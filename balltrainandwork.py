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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define paths
MODEL_PATH = "robotarm.pkl"
CSV_PATH = "robootlog.csv"
BALL_MODEL_PATH = "ball_detector.pth"
DATASET_ROOT = "dataset"

# Dataset configuration
BALL_DATA_CONFIG = {
    'train': {
        'images': os.path.join(DATASET_ROOT, 'train', 'images'),
        'labels': os.path.join(DATASET_ROOT, 'train', 'labels')
    },
    'val': {
        'images': os.path.join(DATASET_ROOT, 'val', 'images'),
        'labels': os.path.join(DATASET_ROOT, 'val', 'labels')
    },
    'img_size': (128, 128),
    'batch_size': 32
}


# Custom Dataset Class
class BallDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = img_name.replace('.jpg', '.txt')

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label (x, y coordinates normalized to [0,1])
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            x, y = map(float, f.read().split())

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([x, y], dtype=torch.float32)


# Enhanced Ball Detection CNN
class BallDetector(nn.Module):
    def __init__(self):
        super(BallDetector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# TD3 Networks (unchanged from original)
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
        x = (x + 1) * 90  # Scale to [0, 180]
        return x


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


# Enhanced TD3 Agent
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

        # Initialize networks
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

        # Replay buffer and counters
        self.memory = deque(maxlen=buffer_size)
        self.train_step_counter = 0
        self.last_action = np.array([90] * action_size)  # Neutral position

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze().numpy()
        self.actor.train()

        if add_noise:
            # Direction-aware noise based on ball position
            ball_x = state[0, 5]  # Normalized ball X position
            directional_noise = np.random.normal(0, 0.2 * (1 - ball_x), size=self.action_size)
            action = action + directional_noise
            action = np.clip(action, 0, 180)

        # Smoothing action transitions
        action = 0.7 * action + 0.3 * self.last_action
        self.last_action = action.copy()

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
        except:
            print("Failed to load model from", path)
            return False


def train_ball_detector():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(BALL_DATA_CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BallDataset(
        BALL_DATA_CONFIG['train']['images'],
        BALL_DATA_CONFIG['train']['labels'],
        transform=transform
    )

    val_dataset = BallDataset(
        BALL_DATA_CONFIG['val']['images'],
        BALL_DATA_CONFIG['val']['labels'],
        transform=transform
    )

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BALL_DATA_CONFIG['batch_size'], shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BALL_DATA_CONFIG['batch_size'], shuffle=False)
    }

    model = BallDetector()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    for epoch in range(50):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), BALL_MODEL_PATH)

            print(f'{phase} Epoch {epoch} Loss: {epoch_loss:.4f}')

    print(f'Best validation loss: {best_loss:.4f}')


def calculate_reward(prev_state, curr_state, action):
    # Position-based reward
    ball_x, ball_y = curr_state[5], curr_state[6]
    servo_pos = curr_state[:5]

    # Distance from center (encourage wide range movement)
    center_dist = np.mean(np.abs(servo_pos - 90)) / 90
    center_reward = 1 - center_dist

    # Ball tracking reward
    tracking_error = np.sqrt((ball_x - 0.5) ** 2 + (ball_y - 0.5) ** 2)
    tracking_reward = 1 - tracking_error

    # Movement penalty to prevent jitter
    movement = np.mean(np.abs(action - prev_state[:5])) / 180
    movement_penalty = -0.1 * movement

    # Hitting reward
    hit_reward = 2.0 if tracking_error < 0.1 else 0

    return center_reward + tracking_reward + movement_penalty + hit_reward


def get_state(servo_angles, ball_pos, prev_ball_pos):
    state = np.zeros(9)
    state[:5] = servo_angles
    if ball_pos is not None:
        state[5:7] = ball_pos
        if prev_ball_pos is not None:
            state[7:] = ball_pos - prev_ball_pos
    return state


def main():
    if not os.path.exists(BALL_MODEL_PATH):
        print("Training ball detector...")
        train_ball_detector()

    ball_detector = BallDetector()
    ball_detector.load_state_dict(torch.load(BALL_MODEL_PATH))
    ball_detector.eval()

    arduino = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)

    agent = TD3Agent(state_size=9, action_size=5)
    cap = cv2.VideoCapture(1)

    prev_ball_pos = None
    current_state = np.array([90] * 5 + [0.5, 0.5, 0, 0])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(BALL_DATA_CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret: continue

        input_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            ball_pos = ball_detector(input_tensor).squeeze().numpy()

        if prev_ball_pos is not None:
            velocity = ball_pos - prev_ball_pos
        else:
            velocity = np.zeros(2)

        current_state[5:7] = ball_pos
        current_state[7:9] = velocity

        action = agent.act(current_state)

        # Dynamic action adjustment
        target_x = ball_pos[0] * 180
        action = 0.7 * action + 0.3 * np.array([target_x] * 5)
        action = np.clip(action, 0, 180)

        command = ",".join(map(str, action.astype(int)))
        arduino.write(f"{command}\n".encode())

        next_state = current_state.copy()
        reward = calculate_reward(current_state, next_state, action)
        agent.remember(current_state, action, reward, next_state, False)
        agent.learn()

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    agent.save(MODEL_PATH)
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()


if __name__ == "__main__":
    main()
