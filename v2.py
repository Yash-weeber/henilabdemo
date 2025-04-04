import cv2
import numpy as np
import serial
import time

# Initialize serial communication with Arduino (change 'COM5' to your port)
arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Give some time for Arduino to initialize

# Initialize webcam
cap = cv2.VideoCapture(2)

# Function to map values from one range to another
def map_range(value, in_min, in_max, out_min, out_max):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# Frame dimensions (to be updated dynamically)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
servo_angle = 90   # Start servo at the middle position (90 degrees)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
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

            # Draw rectangle and center point on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Determine which quadrant the object is in
            if center_x < frame_width // 2 and center_y < frame_height // 2:
                quadrant = "Top-Left"
                servo_angle = map_range(center_x, 0, frame_width // 2, 0, 90)  # Map to left angles
            elif center_x >= frame_width // 2 and center_y < frame_height // 2:
                quadrant = "Top-Right"
                servo_angle = map_range(center_x - frame_width // 2,
                                        0,
                                        frame_width // 2,
                                        90,
                                        180)  # Map to right angles
            elif center_x < frame_width // 2 and center_y >= frame_height // 2:
                quadrant = "Bottom-Left"
                servo_angle = map_range(center_x,
                                        0,
                                        frame_width // 2,
                                        0,
                                        90)  # Map to left angles
            else:
                quadrant = "Bottom-Right"
                servo_angle = map_range(center_x - frame_width // 2,
                                        0,
                                        frame_width // 2,
                                        90,
                                        180)  # Map to right angles

            # Send servo angle to Arduino via serial communication
            arduino.write(f"{servo_angle}\n".encode())
            time.sleep(0.05)  # Add a small delay between writes

            # Display position and angle information on the frame
            cv2.putText(frame,
                        f"Quadrant: {quadrant}, Servo Angle: {servo_angle}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

    # Draw quadrant lines on the frame for visualization
    cv2.line(frame,
             (frame_width // 2, 0),
             (frame_width // 2, frame_height),
             (255, 0, 0),
             thickness=1)
    cv2.line(frame,
             (0, frame_height // 2),
             (frame_width, frame_height // 2),
             (255, 0, 0),
             thickness=1)

    # Display the frames
    cv2.imshow("Yellow Object Tracking", frame)
    cv2.imshow("Mask", mask)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
