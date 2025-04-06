# import cv2
# import numpy as np
# import matlab.engine
# import time
#
# # --- Start MATLAB Engine ---
# eng = matlab.engine.start_matlab()
#
# # --- Initialize webcam ---
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     raise Exception("Could not open video device")
#
# # Get frame dimensions
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#
# def map_image_to_world(x, y, frame_width, frame_height):
#     """
#     Map image coordinates to a world coordinate.
#     This example mapping centers the image at (0,0) with a fixed Z height.
#     Adjust the mapping as needed.
#     """
#     # Map x: left edge -> -1, right edge -> 1
#     world_x = (x - frame_width / 2) / (frame_width / 2)
#     # Map y: top edge -> 1, bottom edge -> -1 (flip y-axis)
#     world_y = -(y - frame_height / 2) / (frame_height / 2)
#     # Fixed z-height; modify as needed
#     world_z = 0.5
#     return [world_x, world_y, world_z]
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip frame horizontally for natural movement
#     frame = cv2.flip(frame, 1)
#
#     # Convert frame to HSV for color detection
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Define yellow color range in HSV
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
#
#     # Create mask to isolate yellow regions
#     mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
#
#     # Morphological operations to reduce noise
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if contours:
#         # Select the largest contour by area
#         largest_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(largest_contour)
#
#         if area > 500:  # Filter out small contours
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             center_x = x + w // 2
#             center_y = y + h // 2
#
#             # Draw bounding rectangle and center point
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
#
#             # Map image coordinates to world coordinates
#             target_world = map_image_to_world(center_x, center_y, frame_width, frame_height)
#             print("Target world coordinates:", target_world)
#
#             # Convert Python list to MATLAB double (as a 1x3 vector)
#             matlab_target = matlab.double(target_world)
#
#             # Call the MATLAB function computeIK to update the simulation
#             try:
#                 result = eng.computeIK(matlab_target, nargout=1)
#                 print("Computed joint angles:", result['jointAngles'])
#             except Exception as e:
#                 print("Error calling MATLAB function:", e)
#
#     # Display the original frame and the mask
#     cv2.imshow("Yellow Object Tracking", frame)
#     cv2.imshow("Mask", mask)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # Slow down loop a little if needed
#     time.sleep(0.05)
#
# # Cleanup resources
# cap.release()
# cv2.destroyAllWindows()
# eng.quit()
######
import cv2
import numpy as np
import matlab.engine
import time
import pandas as pd  # For saving CSV

# --- Start MATLAB Engine ---
eng = matlab.engine.start_matlab()

# --- Initialize webcam ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def map_image_to_world(x, y, frame_width, frame_height):
    """
    Map image coordinates to a world coordinate.
    This example mapping centers the image at (0,0) with a fixed Z height.
    Adjust the mapping as needed.
    """
    # Map x: left edge -> -1, right edge -> 1
    world_x = (x - frame_width / 2) / (frame_width / 2)
    # Map y: top edge -> 1, bottom edge -> -1 (flip y-axis)
    world_y = -(y - frame_height / 2) / (frame_height / 2)
    # Fixed z-height; modify as needed
    world_z = 0.5
    return [world_x, world_y, world_z]


# Create a log to store timestamp, target coordinates, and joint angles
data_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural movement
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create mask to isolate yellow regions
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw bounding rectangle and center point
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Map image coordinates to world coordinates
            target_world = map_image_to_world(center_x, center_y, frame_width, frame_height)
            print("Target world coordinates:", target_world)

            # Convert Python list to MATLAB double (as a 1x3 vector)
            matlab_target = matlab.double(target_world)

            # Call the MATLAB function computeIK to update the simulation
            try:
                result = eng.computeIK(matlab_target, nargout=1)
                joint_angles = list(result['jointAngles'])
                print("Computed joint angles:", joint_angles)

                # Log the result along with a timestamp
                entry = {
                    'timestamp': time.time(),
                    'world_x': target_world[0],
                    'world_y': target_world[1],
                    'world_z': target_world[2]
                }
                # Assuming there are 6 joint angles; adjust as necessary.
                for i, angle in enumerate(joint_angles):
                    entry[f'Joint{i + 1}'] = angle
                data_log.append(entry)
            except Exception as e:
                print("Error calling MATLAB function:", e)

    # Display the original frame and the mask
    cv2.imshow("Yellow Object Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Slow down loop a little if needed
    time.sleep(0.05)

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
eng.quit()

# Save the logged data to a CSV file
df = pd.DataFrame(data_log)
output_csv = 'ik_results.csv'
df.to_csv(output_csv, index=False)
print(f"Logged data saved to {output_csv}")
