# import serial
# import time
# import csv
# import sys
#
# # --------------------
# # User Configurations
# # --------------------
# SERIAL_PORT = "COM10"  # or "COM3" on Windows, etc.
# BAUD_RATE = 115200
# LOG_INTERVAL = 0.5  # seconds, can be changed by user
#
# # Optional: define a “home” angle array in Python as well, if needed
# home_angles = [90, 90, 90, 90, 90]
#
# # Output CSV file
# csv_filename = "robot_arm_log.csv"
#
#
# def main():
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#     except serial.serialutil.SerialException:
#         print(f"Error: Could not open serial port {SERIAL_PORT}")
#         sys.exit(1)
#
#     # Open CSV file in append mode so we don’t overwrite
#     with open(csv_filename, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         # Write a header row
#         csv_writer.writerow([
#             "Timestamp",
#             "Servo1Angle", "Diff1", "Dir1",
#             "Servo2Angle", "Diff2", "Dir2",
#             "Servo3Angle", "Diff3", "Dir3",
#             "Servo4Angle", "Diff4", "Dir4",
#             "Servo5Angle", "Diff5", "Dir5"
#         ])
#
#         print("Logging started. Press Ctrl+C to stop...")
#
#         last_read_time = time.time()
#
#         while True:
#             # Check if enough time has passed since last log
#             current_time = time.time()
#             if (current_time - last_read_time) >= LOG_INTERVAL:
#                 # Attempt to read the last line from the buffer
#                 # If the Arduino is sending data in a loop, we may want
#                 # to flush intermediate lines and only parse the latest line.
#                 line = ""
#                 while ser.in_waiting > 0:
#                     line = ser.readline().decode('utf-8').strip()
#
#                 # If we got a line, parse it
#                 if line:
#                     # Arduino format example:
#                     # servo1Angle,diff1,dir1, servo2Angle,diff2,dir2, ...
#                     # e.g. "90,0,Neutral,100,10,Up/Forward, ... "
#                     fields = line.split(",")
#
#                     if len(fields) == 15:
#                         # fields should be [s1Angle, diff1, dir1, s2Angle, diff2, dir2, ... s5Angle, diff5, dir5]
#                         # Convert numeric fields to integers
#                         # indices: angle @ 0, diff @ 1, dir @ 2, angle @3, diff @4, dir@5, ...
#                         # We'll do a quick parse:
#                         try:
#                             servo1Angle = int(fields[0])
#                             diff1 = int(fields[1])
#                             dir1 = fields[2]
#
#                             servo2Angle = int(fields[3])
#                             diff2 = int(fields[4])
#                             dir2 = fields[5]
#
#                             servo3Angle = int(fields[6])
#                             diff3 = int(fields[7])
#                             dir3 = fields[8]
#
#                             servo4Angle = int(fields[9])
#                             diff4 = int(fields[10])
#                             dir4 = fields[11]
#
#                             servo5Angle = int(fields[12])
#                             diff5 = int(fields[13])
#                             dir5 = fields[14]
#
#                             # Write row to CSV with timestamp
#                             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
#                             csv_writer.writerow([
#                                 timestamp,
#                                 servo1Angle, diff1, dir1,
#                                 servo2Angle, diff2, dir2,
#                                 servo3Angle, diff3, dir3,
#                                 servo4Angle, diff4, dir4,
#                                 servo5Angle, diff5, dir5
#                             ])
#                             csv_file.flush()  # ensure data is saved
#
#                             # Print for debug
#                             print(f"[{timestamp}] S1={servo1Angle}({diff1},{dir1}), "
#                                   f"S2={servo2Angle}({diff2},{dir2}), "
#                                   f"S3={servo3Angle}({diff3},{dir3}), "
#                                   f"S4={servo4Angle}({diff4},{dir4}), "
#                                   f"S5={servo5Angle}({diff5},{dir5})")
#
#                         except ValueError:
#                             # If fields can't be converted to int
#                             print("Parse error on line:", line)
#                     else:
#                         # If it’s not 15 fields, it might be partial or corrupted
#                         print("Unexpected format:", line)
#
#                 last_read_time = current_time
#
#             time.sleep(0.01)  # A small delay so we don’t busy-wait too hard
#
#
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nExiting...")

import serial
import time
import csv
import sys
import argparse

# --------------------
# Default User Configurations
# --------------------
DEFAULT_SERIAL_PORT = "COM10"  # Set to your Arduino's COM port (e.g., "COM10" on Windows or "/dev/ttyACM0" on Linux)
DEFAULT_BAUD_RATE = 115200
DEFAULT_LOG_INTERVAL = 0.5  # Logging interval (seconds)
CSV_FILENAME = "robot_arm_log.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Log robot servo data from Arduino to CSV.")
    parser.add_argument('--port', type=str, default=DEFAULT_SERIAL_PORT, help="Serial port of the Arduino")
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD_RATE, help="Baud rate (default 115200)")
    parser.add_argument('--interval', type=float, default=DEFAULT_LOG_INTERVAL,
                        help="Logging interval in seconds (default 0.5)")
    parser.add_argument('--csv', type=str, default=CSV_FILENAME, help="CSV file name to write data")
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.serialutil.SerialException:
        print(f"Error: Could not open serial port {args.port}")
        sys.exit(1)

    with open(args.csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header row (15 columns: timestamp + 5 groups of 3 values)
        csv_writer.writerow([
            "Timestamp",
            "Servo1Angle", "Diff1", "Dir1",
            "Servo2Angle", "Diff2", "Dir2",
            "Servo3Angle", "Diff3", "Dir3",
            "Servo4Angle", "Diff4", "Dir4",
            "Servo5Angle", "Diff5", "Dir5"
        ])

        print("Logging started. Press Ctrl+C to stop...")
        last_log_time = time.time()

        while True:
            current_time = time.time()
            if (current_time - last_log_time) >= args.interval:
                # Read the most recent complete serial line.
                line = ""
                while ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                    except UnicodeDecodeError:
                        continue

                if line:
                    fields = line.split(",")
                    if len(fields) == 15:
                        try:
                            servo1Angle = int(fields[0])
                            diff1 = int(fields[1])
                            dir1 = fields[2]

                            servo2Angle = int(fields[3])
                            diff2 = int(fields[4])
                            dir2 = fields[5]

                            servo3Angle = int(fields[6])
                            diff3 = int(fields[7])
                            dir3 = fields[8]

                            servo4Angle = int(fields[9])
                            diff4 = int(fields[10])
                            dir4 = fields[11]

                            servo5Angle = int(fields[12])
                            diff5 = int(fields[13])
                            dir5 = fields[14]

                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                            csv_writer.writerow([
                                timestamp,
                                servo1Angle, diff1, dir1,
                                servo2Angle, diff2, dir2,
                                servo3Angle, diff3, dir3,
                                servo4Angle, diff4, dir4,
                                servo5Angle, diff5, dir5
                            ])
                            csv_file.flush()

                            print(f"[{timestamp}] S1={servo1Angle} ({diff1}, {dir1}), " +
                                  f"S2={servo2Angle} ({diff2}, {dir2}), " +
                                  f"S3={servo3Angle} ({diff3}, {dir3}), " +
                                  f"S4={servo4Angle} ({diff4}, {dir4}), " +
                                  f"S5={servo5Angle} ({diff5}, {dir5})")
                        except ValueError:
                            print("Parse error on line:", line)
                    else:
                        print("Unexpected format:", line)

                last_log_time = current_time

            time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
