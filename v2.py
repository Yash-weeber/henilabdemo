import cv2
import numpy as np
import serial
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle
from datetime import datetime

# Define paths
MODEL_PATH = "robot_arm_rl_model1.pkl"
CSV_PATH = "robotlog.csv"


# Neural Network Architecture for Actor (Policy) Network
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # Scale from [-1, 1] to [0, 180] for servo angles
        x = (x + 1) * 90
        return x


# Neural Network Architecture for Critic (Value) Network
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# TD3 Agent implementation
class TD3Agent:
    def __init__(self, state_size, action_size, hidden_size=128, buffer_size=10000,
                 batch_size=64, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Actor and critic networks
        self.actor = ActorNetwork(state_size, action_size, hidden_size)
        self.actor_target = ActorNetwork(state_size, action_size, hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic1 = CriticNetwork(state_size, action_size, hidden_size)
        self.critic2 = CriticNetwork(state_size, action_size, hidden_size)
        self.critic1_target = CriticNetwork(state_size, action_size, hidden_size)
        self.critic2_target = CriticNetwork(state_size, action_size, hidden_size)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.train_step_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze().numpy()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=self.action_size)
            action = action + noise
            action = np.clip(action, 0, 180)  # Clip to valid servo range

        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.FloatTensor(np.vstack(dones))

        # Update critic networks
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states)
            next_actions = torch.clamp(next_actions + noise, 0, 180)

            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)

            target_q = rewards + (1 - dones) * self.gamma * q_target

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        self.train_step_counter += 1
        if self.train_step_counter % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self._update_targets()

    def _update_targets(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        model_data = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.actor.load_state_dict(model_data['actor'])
            self.actor_target.load_state_dict(model_data['actor_target'])
            self.critic1.load_state_dict(model_data['critic1'])
            self.critic1_target.load_state_dict(model_data['critic1_target'])
            self.critic2.load_state_dict(model_data['critic2'])
            self.critic2_target.load_state_dict(model_data['critic2_target'])

            print("Successfully loaded model from", path)
            return True
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
            return False


# Function to load and preprocess CSV data - Adapted for the dataset format
def load_csv_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found!")
        return pd.DataFrame()

    try:
        # Load the dataset with the specific format shown in the data
        data = pd.read_csv(csv_path)
        print(f"Loaded CSV data with {len(data)} records and columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()


# Function to extract training data from CSV - Adapted for the dataset format
def extract_training_data(data):
    if data.empty:
        print("No data available for training")
        return [], []

    states = []
    actions = []

    try:
        # Data format from the sample data shown (using direct column indices)
        # Check if we have the expected column headers
        expected_cols = ['timestamp', 'base_angle', 'base_diff', 'base_dir',
                         'shoulder_angle', 'shoulder_diff', 'shoulder_dir',
                         'elbow_angle', 'elbow_diff', 'elbow_dir',
                         'wrist_angle', 'wrist_diff', 'wrist_dir',
                         'gripper_angle']

        # Get servo angle columns
        cols = data.columns.tolist()
        angle_cols = [c for c in cols if c.endswith('_angle')]

        if len(angle_cols) >= 5:  # We need at least 5 angle columns
            for i in range(len(data) - 1):
                # Current state (servo angles) - using only angle columns
                current_angles = []
                for col in angle_cols[:5]:  # Use first 5 angle columns
                    current_angles.append(data.iloc[i][col])

                # Next state as the action to take
                next_angles = []
                for col in angle_cols[:5]:  # Use first 5 angle columns
                    next_angles.append(data.iloc[i + 1][col])

                states.append(np.array(current_angles, dtype=np.float32))
                actions.append(np.array(next_angles, dtype=np.float32))
        else:
            # Fallback to direct column indices - this matches the format in the data
            for i in range(len(data) - 1):
                # Get the angles from the current row using column indices:
                # Assuming columns: base_angle, shoulder_angle, elbow_angle, wrist_angle, gripper_angle
                state = np.array([
                    data.iloc[i, 1],  # base_angle (column index 1)
                    data.iloc[i, 4],  # shoulder_angle (column index 4)
                    data.iloc[i, 7],  # elbow_angle (column index 7)
                    data.iloc[i, 10],  # wrist_angle (column index 10)
                    data.iloc[i, 13]  # gripper_angle (column index 13)
                ], dtype=np.float32)

                # Get the angles from the next row
                action = np.array([
                    data.iloc[i + 1, 1],  # base_angle
                    data.iloc[i + 1, 4],  # shoulder_angle
                    data.iloc[i + 1, 7],  # elbow_angle
                    data.iloc[i + 1, 10],  # wrist_angle
                    data.iloc[i + 1, 13]  # gripper_angle
                ], dtype=np.float32)

                states.append(state)
                actions.append(action)

        print(f"Extracted {len(states)} state-action pairs for training")
        return states, actions

    except Exception as e:
        print(f"Error extracting training data: {e}")
        return [], []


# Function to setup and train agent
def setup_agent(csv_path, model_path):
    # Load CSV data
    data = load_csv_data(csv_path)

    # Define state and action sizes
    # State = 5 servo angles + 2 object position (x, y)
    state_size = 7
    # Action = 5 servo angles
    action_size = 5

    # Create agent
    agent = TD3Agent(state_size, action_size)

    # Try to load existing model
    if os.path.exists(model_path) and agent.load(model_path):
        print("Using pre-trained model")
        return agent

    # No existing model, train new one if data available
    if not data.empty:
        print("Training new model from CSV data")
        states, actions = extract_training_data(data)

        if states and actions:
            # Pre-fill agent memory with state-action pairs
            for i in range(len(states)):
                state = np.append(states[i], [0, 0])  # Append zeros for object position
                action = actions[i]

                # For reward, we're assuming actions that move towards a goal are positive
                reward = 0.1  # Small positive reward for all actions as baseline
                next_state = np.append(actions[i], [0, 0])
                done = False

                agent.remember(state, action, reward, next_state, done)

            # Train the agent
            print("Starting training...")
            epochs = min(100000, len(states) * 5)  # Scale training based on data size
            for epoch in range(epochs):
                agent.learn()
                if epoch % 10 == 0:
                    print(f"Completed training epoch {epoch}/{epochs}")

            # Save the trained model
            agent.save(model_path)
            print(f"Model saved to {model_path}")

    return agent


# Helper function to get direction text based on difference value
def get_direction(diff):
    if diff == 0:
        return "center"
    elif diff > 0:
        return "right" if abs(diff) > 45 else "up"
    else:  # diff < 0
        return "left" if abs(diff) > 45 else "down"


# Function to log data to CSV that matches the format of the dataset
def log_data(csv_path, servo_angles, object_pos=None, hit_direction=None):
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")

    # Home position for calculating differences (90° is considered center)
    home_positions = [90, 90, 90, 90, 90]

    # Initialize data dictionary
    data = {
        'timestamp': timestamp,
        'base_angle': int(servo_angles[0]),
        'base_diff': int(servo_angles[0] - home_positions[0]),
        'base_dir': get_direction(int(servo_angles[0] - home_positions[0])),
        'shoulder_angle': int(servo_angles[1]),
        'shoulder_diff': int(servo_angles[1] - home_positions[1]),
        'shoulder_dir': get_direction(int(servo_angles[1] - home_positions[1])),
        'elbow_angle': int(servo_angles[2]),
        'elbow_diff': int(servo_angles[2] - home_positions[2]),
        'elbow_dir': get_direction(int(servo_angles[2] - home_positions[2])),
        'wrist_angle': int(servo_angles[3]),
        'wrist_diff': int(servo_angles[3] - home_positions[3]),
        'wrist_dir': get_direction(int(servo_angles[3] - home_positions[3])),
        'gripper_angle': int(servo_angles[4])
    }

    # Add object position if available
    if object_pos:
        data['object_x'] = object_pos[0]
        data['object_y'] = object_pos[1]

    # Add hit direction if available
    if hit_direction:
        data['hit_direction'] = hit_direction

    # Convert to DataFrame
    df_row = pd.DataFrame([data])

    # Append to CSV
    try:
        if os.path.exists(csv_path):
            df_row.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error logging data: {e}")


# Function to calculate reward based on object movement
def calculate_reward(object_detected, prev_pos, curr_pos, hit_direction='away'):
    if not object_detected or prev_pos is None or curr_pos is None:
        return 0.0

    # Calculate movement vector
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    # Magnitude of movement
    movement = np.sqrt(dx * dx + dy * dy)

    # If object barely moved, small positive reward for detection
    if movement < 0.01:
        return 0.1

    # For object moving away (positive X direction)
    if hit_direction == 'away' and dx > 0:
        return 1.0 * movement  # Reward proportional to movement

    # For object moving back (negative X direction)
    if hit_direction == 'back' and dx < 0:
        return 1.0 * movement

    # Object moved in wrong direction
    return -0.5 * movement  # Penalty proportional to movement


# Main function
def main():
    # Initialize serial communication with Arduino
    try:
        arduino = serial.Serial('COM5', 9600, timeout=1)
        time.sleep(2)  # Give time for Arduino to initialize
        print("Arduino connected successfully")
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        print("Running in simulation mode without hardware")
        arduino = None

    # Setup agent
    agent = setup_agent(CSV_PATH, MODEL_PATH)

    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam. Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera. Exiting.")
            if arduino:
                arduino.close()
            return

    # Frame dimensions
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Initialize state
    # [base, shoulder, elbow, wrist, gripper, obj_x, obj_y]
    current_state = np.array([90, 90, 90, 90, 90, 0, 0], dtype=np.float32)

    # Object tracking variables
    previous_position = None
    object_detected = False
    hit_count = 0

    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip the frame horizontally for intuitive movement
            frame = cv2.flip(frame, 1)

            # Convert frame to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define yellow color range in HSV
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Create a mask for detecting yellow color
            mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

            # Reduce noise using morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours of the yellow objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_position = None

            if contours:
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 500:  # Only track if the area is large enough to avoid noise
                    # Get bounding rectangle of the largest contour
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Calculate the center of the yellow object
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Normalize object position to [0, 1]
                    norm_x = center_x / frame_width
                    norm_y = center_y / frame_height

                    # Update current position
                    current_position = (norm_x, norm_y)
                    object_detected = True

                    # Update object position in state
                    current_state[5:7] = [norm_x, norm_y]

                    # Calculate reward based on object movement
                    reward = calculate_reward(object_detected, previous_position, current_position)

                    # Get action from agent (servo angles)
                    action = agent.act(current_state)

                    # Send servo angles to Arduino
                    if arduino:
                        command = ",".join([str(int(angle)) for angle in action])
                        arduino.write(f"{command}\n".encode())

                    # Draw bounding box and display info
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Display joint names and angles on frame
                    joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
                    for i, (name, angle) in enumerate(zip(joint_names, action)):
                        diff = int(angle - 90)
                        direction = get_direction(diff)
                        angle_text = f"{name}: {int(angle)}° ({diff}, {direction})"
                        cv2.putText(frame, angle_text, (10, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Display reward on frame
                    reward_text = f"Reward: {reward:.2f}"
                    cv2.putText(frame, reward_text, (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Check if object was hit (significant movement)
                    if previous_position is not None:
                        move_dist = np.sqrt((norm_x - previous_position[0]) ** 2 +
                                            (norm_y - previous_position[1]) ** 2)
                        if move_dist > 0.05:  # Threshold for significant movement
                            hit_count += 1
                            hit_direction = "away" if norm_x > previous_position[0] else "back"

                            # Log the hit to CSV
                            log_data("robot_arm_hits.csv", action, (norm_x, norm_y), hit_direction)

                            # Display hit info
                            hit_text = f"Hit #{hit_count}: {hit_direction}"
                            cv2.putText(frame, hit_text, (10, 210),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Update previous position
                    previous_position = current_position

                    # Get current servo positions from Arduino feedback
                    if arduino and arduino.in_waiting > 0:
                        feedback = arduino.readline().decode().strip()
                        if feedback.startswith("Current angles:"):
                            try:
                                servo_parts = feedback.split(':')[1].strip().split(',')
                                if len(servo_parts) == 5:
                                    for i in range(5):
                                        current_state[i] = float(servo_parts[i])
                            except Exception as e:
                                print(f"Error parsing Arduino feedback: {e}")

                    # Store experience for learning
                    next_state = current_state.copy()
                    if reward != 0:  # Only learn from meaningful experiences
                        agent.remember(current_state.copy(), action, reward, next_state, False)

                        # Periodically train the agent and save the model
                        if len(agent.memory) >= agent.batch_size:
                            agent.learn()
                            if len(agent.memory) % 100 == 0:
                                agent.save(MODEL_PATH)

                    # Log data periodically to the main CSV file
                    if random.random() < 0.05:  # Log ~5% of frames
                        log_data(CSV_PATH, action)

            # Display the frames
            cv2.imshow("Object Tracking", frame)
            cv2.imshow("Mask", mask)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Save the model before exiting
        agent.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()


if __name__ == "__main__":
    main()
