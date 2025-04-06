# import cv2
# import numpy as np
# import matlab.engine
# import time
#
# # Start the MATLAB engine.
# eng = matlab.engine.start_matlab()
#
# # Initialize webcam.
# cap = cv2.VideoCapture(1)
#
#
# # Map image coordinates (from the webcam) to robot workspace coordinates.
# def map_image_to_workspace(x, y, frame_width, frame_height):
#     # Example mapping:
#     # - Center of image maps to [0, 0, 0.5] (in meters)
#     # - x in image maps to workspace x in range [-0.5, 0.5]
#     # - y in image maps to workspace y in range [-0.5, 0.5]
#     workspace_x = (x - frame_width / 2) / (frame_width / 2) * 0.5
#     workspace_y = -(y - frame_height / 2) / (frame_height / 2) * 0.5  # Invert y if needed.
#     workspace_z = 0.5  # Fixed height; adjust based on your application.
#     return [workspace_x, workspace_y, workspace_z]
#
#
# # Get frame dimensions.
# ret, frame = cap.read()
# if not ret:
#     raise Exception("Could not read from webcam.")
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Get the initial (home) configuration from MATLAB.
# # The MATLAB function homeConfiguration requires the URDF file path.
# # Adjust the path below if needed.
# current_config = eng.homeConfiguration("E:/ras/henilabdemo/my_pro600.urdf", nargout=1)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip the frame horizontally.
#     frame = cv2.flip(frame, 1)
#
#     # Convert frame to HSV.
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Define yellow color range in HSV.
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
#
#     # Create a mask for yellow color.
#     mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
#
#     # Use morphological operations to reduce noise.
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # Find contours in the mask.
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if contours:
#         # Choose the largest contour.
#         largest_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(largest_contour)
#
#         if area > 500:  # Filter out noise.
#             # Get bounding rectangle and center of the contour.
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             center_x = x + w // 2
#             center_y = y + h // 2
#
#             # Draw visualization on the frame.
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
#
#             # Map the image center to a workspace coordinate.
#             target_pose = map_image_to_workspace(center_x, center_y, frame_width, frame_height)
#
#             # Convert target_pose to a MATLAB double array.
#             matlab_target = matlab.double(target_pose)
#
#             # Call the MATLAB function to update the simulation.
#             # Pass the target pose and the current configuration.
#             new_config = eng.update_robot_sim(matlab_target, current_config, nargout=1)
#             current_config = new_config  # Update the current configuration.
#
#     # Display the camera feed and the mask.
#     cv2.imshow("Yellow Object Tracking", frame)
#     cv2.imshow("Mask", mask)
#
#     # Break on 'q' key press.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources.
# cap.release()
# cv2.destroyAllWindows()
# eng.quit()




######
import cv2
import numpy as np
import matlab.engine
import time

# Start the MATLAB engine.
eng = matlab.engine.start_matlab()

# Initialize webcam.
cap = cv2.VideoCapture(1)


# Map image coordinates (from the webcam) to robot workspace coordinates.
def map_image_to_workspace(x, y, frame_width, frame_height):
    # Example mapping:
    # - Center of image maps to [0, 0, 0.5] (in meters)
    # - x in image maps to workspace x in range [-0.5, 0.5]
    # - y in image maps to workspace y in range [-0.5, 0.5]
    workspace_x = (x - frame_width / 2) / (frame_width / 2) * 0.5
    workspace_y = -(y - frame_height / 2) / (frame_height / 2) * 0.5  # Invert y if needed.
    workspace_z = 0.5  # Fixed height; adjust based on your application.
    return [workspace_x, workspace_y, workspace_z]


# Get frame dimensions.
ret, frame = cap.read()
if not ret:
    raise Exception("Could not read from webcam.")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the initial (home) configuration from MATLAB by calling our helper.
current_config = eng.initRobotSim("E:\\ras\\henilabdemo\\my_pro600.urdf", nargout=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally.
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV.
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow color.
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Use morphological operations to reduce noise.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Choose the largest contour.
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:  # Filter out noise.
            # Get bounding rectangle and center of the contour.
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw visualization on the frame.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Map the image center to a workspace coordinate.
            target_pose = map_image_to_workspace(center_x, center_y, frame_width, frame_height)

            # Convert target_pose to a MATLAB double array.
            matlab_target = matlab.double(target_pose)

            # Call the MATLAB function to update the simulation.
            # Pass the target pose and the current configuration.
            new_config = eng.update_robot_sim(matlab_target, current_config, nargout=1)
            current_config = new_config  # Update the current configuration.

    # Display the camera feed and the mask.
    cv2.imshow("Yellow Object Tracking", frame)
    cv2.imshow("Mask", mask)

    # Break on 'q' key press.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
eng.quit()
