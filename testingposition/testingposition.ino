// /*******************************************************
//  * ARDUINO SIDE
//  * Reads 5 knobs (or servo feedback lines) for 5 DOF
//  * and sends the angles + difference from home via Serial.
//  *******************************************************/

// #include <Arduino.h>
// #include <Servo.h>
// #define S1 A0
// #define S2 A1
// #define S3 A2
// #define S4 A3
// #define S5 A4

// // If you have standard servo library usage, you might do:
// // #include <Servo.h>
// // But for just reading knobs, no servo attach is strictly needed.

// // Define which analog pins your 6-ch knob shield uses.
// // (This is just an example—check the actual pins on your shield)
// #define KNOB1_PIN S1
// #define KNOB2_PIN S2
// #define KNOB3_PIN S3
// #define KNOB4_PIN S4
// #define KNOB5_PIN S5
// // If you have a 6th knob, A5 might be used, but we only have 5 servo DOFs.

// // Home angles for each servo (adjust to your real “home”)
// int homeServo1 = 90;
// int homeServo2 = 90;
// int homeServo3 = 90;
// int homeServo4 = 90;
// int homeServo5 = 90;

// void setup() {
//   Serial.begin(115200); // Make sure Python matches this
// }

// void loop() {
//   // 1) Read analog inputs (0 to 1023)
//   int raw1 = analogRead(KNOB1_PIN);
//   int raw2 = analogRead(KNOB2_PIN);
//   int raw3 = analogRead(KNOB3_PIN);
//   int raw4 = analogRead(KNOB4_PIN);
//   int raw5 = analogRead(KNOB5_PIN);

//   // 2) Map raw analog (0–1023) to approximate servo angle (0–180)
//   int servo1Angle = map(raw1, 0, 1023, 0, 180);
//   int servo2Angle = map(raw2, 0, 1023, 0, 180);
//   int servo3Angle = map(raw3, 0, 1023, 0, 180);
//   int servo4Angle = map(raw4, 0, 1023, 0, 180);
//   int servo5Angle = map(raw5, 0, 1023, 0, 180);

//   // 3) Calculate difference from home
//   int diff1 = servo1Angle - homeServo1;
//   int diff2 = servo2Angle - homeServo2;
//   int diff3 = servo3Angle - homeServo3;
//   int diff4 = servo4Angle - homeServo4;
//   int diff5 = servo5Angle - homeServo5;

//   // 4) (Optional) Convert difference to direction strings for demonstration
//   //    For a real robot, you might do something more sophisticated
//   String dir1 = (diff1 > 0) ? "Up/Forward" : ((diff1 < 0) ? "Down/Backward" : "Neutral");
//   String dir2 = (diff2 > 0) ? "Up/Forward" : ((diff2 < 0) ? "Down/Backward" : "Neutral");
//   String dir3 = (diff3 > 0) ? "Up/Forward" : ((diff3 < 0) ? "Down/Backward" : "Neutral");
//   String dir4 = (diff4 > 0) ? "Up/Forward" : ((diff4 < 0) ? "Down/Backward" : "Neutral");
//   String dir5 = (diff5 > 0) ? "Up/Forward" : ((diff5 < 0) ? "Down/Backward" : "Neutral");

//   // 5) Print angles and directions in a simple comma-separated format
//   // Format:  servo1Angle,diff1,dir1, servo2Angle,diff2,dir2, ...
//   // Each line is a “packet” for Python
//   Serial.print(servo1Angle);
//   Serial.print(",");
//   Serial.print(diff1);
//   Serial.print(",");
//   Serial.print(dir1);
//   Serial.print(",");

//   Serial.print(servo2Angle);
//   Serial.print(",");
//   Serial.print(diff2);
//   Serial.print(",");
//   Serial.print(dir2);
//   Serial.print(",");

//   Serial.print(servo3Angle);
//   Serial.print(",");
//   Serial.print(diff3);
//   Serial.print(",");
//   Serial.print(dir3);
//   Serial.print(",");

//   Serial.print(servo4Angle);
//   Serial.print(",");
//   Serial.print(diff4);
//   Serial.print(",");
//   Serial.print(dir4);
//   Serial.print(",");

//   Serial.print(servo5Angle);
//   Serial.print(",");
//   Serial.print(diff5);
//   Serial.print(",");
//   Serial.print(dir5);

//   Serial.println();  // End of line

//   // Optionally delay a bit here, or let the loop run as fast as it can
//   delay(50); // For example, 50ms
// }

// #include <Arduino.h>

// //-----------------------------------------------------------------
// // Define digital pins to which the PWM signals (from channels S1-S5)
// // are wired. Update these pin assignments based on your wiring.
// #define S1_PIN 2
// #define S2_PIN 3
// #define S3_PIN 4
// #define S4_PIN 5
// #define S5_PIN 6

// //-----------------------------------------------------------------
// // Define the "home" (neutral) positions for each servo (in degrees).
// int homeServo1 = 90;
// int homeServo2 = 90;
// int homeServo3 = 90;
// int homeServo4 = 90;
// int homeServo5 = 90;

// //-----------------------------------------------------------------
// // Helper function to map measured PWM pulse width (in microseconds)
// // to an angle from 0 to 180 degrees.
// //
// // Assumption: A pulse width of 1000 µs = 0° and 2000 µs = 180°.
// // You might need to adjust these numbers based on your servo's calibration.
// int mapPulseToAngle(unsigned long pulseWidth) {
//   // If the pulse is out-of-range, you can still constrain it.
//   return constrain(map(pulseWidth, 1000, 2000, 0, 180), 0, 180);
// }

// void setup() {
//   Serial.begin(115200);

//   // Set the designated pins as inputs.
//   pinMode(S1_PIN, INPUT);
//   pinMode(S2_PIN, INPUT);
//   pinMode(S3_PIN, INPUT);
//   pinMode(S4_PIN, INPUT);
//   pinMode(S5_PIN, INPUT);
// }

// void loop() {
//   // Use pulseIn() to measure the duration (in microseconds) of each PWM signal.
//   // The timeout parameter (here 25000 microseconds) should be longer than the PWM period.
//   unsigned long pulse1 = pulseIn(S1_PIN, HIGH, 25000);
//   unsigned long pulse2 = pulseIn(S2_PIN, HIGH, 25000);
//   unsigned long pulse3 = pulseIn(S3_PIN, HIGH, 25000);
//   unsigned long pulse4 = pulseIn(S4_PIN, HIGH, 25000);
//   unsigned long pulse5 = pulseIn(S5_PIN, HIGH, 25000);
  
//   // Convert the measured pulse widths to angles.
//   int servo1Angle = mapPulseToAngle(pulse1);
//   int servo2Angle = mapPulseToAngle(pulse2);
//   int servo3Angle = mapPulseToAngle(pulse3);
//   int servo4Angle = mapPulseToAngle(pulse4);
//   int servo5Angle = mapPulseToAngle(pulse5);

//   // Calculate the movement from the home (neutral) position.
//   int diff1 = servo1Angle - homeServo1;
//   int diff2 = servo2Angle - homeServo2;
//   int diff3 = servo3Angle - homeServo3;
//   int diff4 = servo4Angle - homeServo4;
//   int diff5 = servo5Angle - homeServo5;

//   // Print the data as comma-separated values.
//   // Format: servoAngle, difference-from-home for each channel.
//   Serial.print(servo1Angle); Serial.print(",");
//   Serial.print(diff1);       Serial.print(",");

//   Serial.print(servo2Angle); Serial.print(",");
//   Serial.print(diff2);       Serial.print(",");

//   Serial.print(servo3Angle); Serial.print(",");
//   Serial.print(diff3);       Serial.print(",");

//   Serial.print(servo4Angle); Serial.print(",");
//   Serial.print(diff4);       Serial.print(",");

//   Serial.print(servo5Angle); Serial.print(",");
//   Serial.println(diff5);

//   // Delay briefly – adjust this delay based on how frequently you want to sample.
//   delay(50);
// }

#include <Arduino.h>

//-----------------------------------------------------------------
// Define digital pins to which the PWM signals (from channels S1-S5)
// are wired. Update these pin assignments based on your wiring.
#define S1_PIN 2
#define S2_PIN 3
#define S3_PIN 4
#define S4_PIN 5
#define S5_PIN 6

//-----------------------------------------------------------------
// Define the "home" (neutral) positions for each servo (in degrees).
int homeServo1 = 90;
int homeServo2 = 90;
int homeServo3 = 90;
int homeServo4 = 90;
int homeServo5 = 90;

//-----------------------------------------------------------------
// Helper function to map measured PWM pulse width (in microseconds)
// to an angle from 0 to 180 degrees.
//
// Assumption: A pulse width of 1000 µs = 0° and 2000 µs = 180°.
// Adjust these numbers based on your servo's calibration.
int mapPulseToAngle(unsigned long pulseWidth) {
  return constrain(map(pulseWidth, 1000, 2000, 0, 180), 0, 180);
}

//-----------------------------------------------------------------
// Helper function to generate a direction string based on difference
// from home position. Adjust the strings as needed for your robot.
String getDirection(int diff) {
  if (diff > 0) {
    return "Up/Forward";
  } else if (diff < 0) {
    return "Down/Backward";
  } else {
    return "Neutral";
  }
}

void setup() {
  Serial.begin(115200);

  // Set the designated pins as inputs.
  pinMode(S1_PIN, INPUT);
  pinMode(S2_PIN, INPUT);
  pinMode(S3_PIN, INPUT);
  pinMode(S4_PIN, INPUT);
  pinMode(S5_PIN, INPUT);
}

void loop() {
  // Use pulseIn() to measure the duration (in microseconds) of each PWM signal.
  // The timeout (25000 microseconds) should be longer than the PWM period.
  unsigned long pulse1 = pulseIn(S1_PIN, HIGH, 25000);
  unsigned long pulse2 = pulseIn(S2_PIN, HIGH, 25000);
  unsigned long pulse3 = pulseIn(S3_PIN, HIGH, 25000);
  unsigned long pulse4 = pulseIn(S4_PIN, HIGH, 25000);
  unsigned long pulse5 = pulseIn(S5_PIN, HIGH, 25000);
  
  // Convert the measured pulse widths to angles.
  int servo1Angle = mapPulseToAngle(pulse1);
  int servo2Angle = mapPulseToAngle(pulse2);
  int servo3Angle = mapPulseToAngle(pulse3);
  int servo4Angle = mapPulseToAngle(pulse4);
  int servo5Angle = mapPulseToAngle(pulse5);

  // Calculate the movement from the home (neutral) position.
  int diff1 = servo1Angle - homeServo1;
  int diff2 = servo2Angle - homeServo2;
  int diff3 = servo3Angle - homeServo3;
  int diff4 = servo4Angle - homeServo4;
  int diff5 = servo5Angle - homeServo5;

  // Generate a direction string based on the movement.
  String dir1 = getDirection(diff1);
  String dir2 = getDirection(diff2);
  String dir3 = getDirection(diff3);
  String dir4 = getDirection(diff4);
  String dir5 = getDirection(diff5);

  // Print the data as comma-separated values (15 fields).
  // The format is: servoAngle,diff,dir for each servo channel.
  Serial.print(servo1Angle); Serial.print(",");
  Serial.print(diff1);       Serial.print(",");
  Serial.print(dir1);        Serial.print(",");

  Serial.print(servo2Angle); Serial.print(",");
  Serial.print(diff2);       Serial.print(",");
  Serial.print(dir2);        Serial.print(",");

  Serial.print(servo3Angle); Serial.print(",");
  Serial.print(diff3);       Serial.print(",");
  Serial.print(dir3);        Serial.print(",");

  Serial.print(servo4Angle); Serial.print(",");
  Serial.print(diff4);       Serial.print(",");
  Serial.print(dir4);        Serial.print(",");

  Serial.print(servo5Angle); Serial.print(",");
  Serial.print(diff5);       Serial.print(",");
  Serial.println(dir5);

  // Delay briefly – adjust this delay based on how frequently you want to sample.
  delay(50);
}
