# Real-Time Object Tracking and MATLAB Simulation Integration (Python)
import cv2
import numpy as np
import matlab.engine
import threading
import time
import queue
from threading import Lock
matlab_lock = Lock()
# Initialize MATLAB Engine
eng = matlab.engine.start_matlab()
eng.addpath(r'E:\ras\henilabdemo', nargout=0)
# Shared queue for position data
position_queue = queue.Queue(maxsize=1)


def matlab_simulation_loop():
    global matlab_lock
    with matlab_lock:

        # Initialize robot model
        robot = eng.importrobot("E:/ras/henilabdemo/my_pro600.urdf", nargout=1)
        config = eng.homeConfiguration(robot)
        config = [eng.struct(conf) for conf in eng.cell(config).flatten()]
        prev_angles = eng.cell(config)
        fig = eng.figure(1, nargout=1)
        eng.show(robot, config, nargout=0)

        # Simulation parameters
        eng.axis(matlab.double([-1, 1, -1, 1, 0, 1.5]), nargout=0)
        eng.view(matlab.double([-100, 90]), nargout=0)
        eng.grid('on', nargout=0)
        eng.hold('on', nargout=0)

        # IK parameters
        euler_angles = matlab.double([178, 0.0, 0])
        gik = eng.generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'pose'}, nargout=1)
        eng.set(gik, 'SolverParameters', matlab.double([500]), 'MaxIterations', nargout=0)

        prev_angles = eng.cell(config)
        # Add in MATLAB loop
        last_update = time.time()


        while True:
            if time.time() - last_update > 0.033:
                try:
                    last_update = time.time()
                    # Get latest target position
                    target_pos = position_queue.get_nowait()
                    matlab_pos = matlab.double(target_pos, is_complex=False)

                    # Compute IK
                    tform = eng.eul2tform(eng.deg2rad(euler_angles), 'XYZ', nargout=1)
                    eng.set(tform, 'Subs', matlab.double([1, 3]), ':', matlab_pos, nargout=0)
                    pose_constraint = eng.constraintPoseTarget('link6', nargout=1)
                    eng.set(pose_constraint, 'TargetTransform', tform, nargout=0)

                    # Solve IK
                    config_soln = eng.cell(config_soln)
                    eng.show(robot, config_soln, 'PreservePlot', False, nargout=0)
                    prev_angles = config_soln
                    eng.drawnow(nargout=0)

                except queue.Empty:
                    time.sleep(0.01)
                except KeyboardInterrupt:
                    break


# Start MATLAB simulation thread
sim_thread = threading.Thread(target=matlab_simulation_loop)
sim_thread.daemon = True
sim_thread.start()

# Computer Vision Configuration
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Camera calibration parameters (example values - should be calibrated for your setup)
focal_length = 17857   #800  # pixels
known_z = 0.3  # meters (assumed working distance)
sensor_width = 5.6e-3  # 5.6mm for S24 Ultra front camera


# Add before main loop
class ObjectKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def update(self, x, y):
        self.kf.correct(np.array([[x], [y]], np.float32))
        prediction = self.kf.predict()
        return prediction[0][0], prediction[1][0]
kf = ObjectKalmanFilter()

# def pixel_to_world(x, y):
#     """Convert pixel coordinates to robot workspace coordinates"""
#     # Simple perspective projection (adjust based on your camera setup)
#     scale_factor = known_z / focal_length
#     world_x = (x - frame_width / 2) * scale_factor
#     world_y = (y - frame_height / 2) * scale_factor
#     return (world_x, world_y, known_z)
# Add to Python tracking code
# Camera sensor width in meters (adjust for your hardware)
# def pixel_to_world(x, y, w):  # Add width parameter
#     object_width = w / frame_width
#     depth = (known_z * focal_length) / (object_width * sensor_width)
#     x_norm = (x - frame_width/2) * depth / focal_length
#     y_norm = (y - frame_height/2) * depth / focal_length
#     return (x_norm, y_norm, depth)

def pixel_to_world(x, y, w):
    object_width = w / frame_width
    depth = (known_z * focal_length) / (object_width * sensor_width)
    x_norm = (x - frame_width/2) * depth / focal_length
    y_norm = (y - frame_height/2) * depth / focal_length
    return (x_norm, y_norm, depth)

# Object tracking parameters
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = kf.update(x + w // 2, y + h // 2)
                target_pos = pixel_to_world(center_x, center_y, w)  # Add width parameter

                # if area > 500:
            #     x, y, w, h = cv2.boundingRect(largest_contour)
            #     # center_x = x + w // 2
            #     # center_y = y + h // 2
            #     raw_x = x + w // 2
            #     raw_y = y + h // 2
            #     center_x, center_y = kf.update(raw_x, raw_y)

                # Convert to robot coordinates
                #target_pos = pixel_to_world(center_x, center_y)

                # Update position queue
                # Before sending to MATLAB:
                target_pos = [float(coord) for coord in target_pos]
                matlab_pos = matlab.double(target_pos, is_complex=False)

                try:
                    position_queue.put_nowait(target_pos)
                except queue.Full:
                    pass

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    eng.quit()
    eng = matlab.engine.start_matlab()  # Add engine restart

