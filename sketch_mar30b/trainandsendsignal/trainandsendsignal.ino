#include <Servo.h>

// Create Servo objects for each motor
Servo servo1; // Base
Servo servo2; // Shoulder
Servo servo3; // Elbow
Servo servo4; // Wrist
Servo servo5; // Gripper

// Servo pins - adjust as needed for your setup
const int SERVO1_PIN = 3;  // Base servo
const int SERVO2_PIN = 5;  // Shoulder servo
const int SERVO3_PIN = 6;  // Elbow servo
const int SERVO4_PIN = 9;  // Wrist servo
const int SERVO5_PIN = 10; // Gripper servo

// Optional: Ultrasonic sensor pins (commented out for future use)
// const int TRIG_PIN = 11;
// const int ECHO_PIN = 12;

// Variables to store servo angles
int servo1Angle = 90;
int servo2Angle = 90;
int servo3Angle = 90;
int servo4Angle = 90;
int servo5Angle = 90;

// Home position angles
const int HOME_SERVO1 = 90;
const int HOME_SERVO2 = 90;
const int HOME_SERVO3 = 90;
const int HOME_SERVO4 = 90;
const int HOME_SERVO5 = 90;

// Timing variables
unsigned long lastFeedbackTime = 0;
const unsigned long FEEDBACK_INTERVAL = 100; // Send feedback every 100ms

// Function to smoothly move servo from current position to target position
void moveServoSmooth(Servo &servo, int &currentAngle, int targetAngle, int step = 5, int delayMs = 15) {
  if (currentAngle < targetAngle) {
    for (int angle = currentAngle; angle <= targetAngle; angle += step) {
      servo.write(angle);
      currentAngle = angle;
      delay(delayMs);
    }
  } else if (currentAngle > targetAngle) {
    for (int angle = currentAngle; angle >= targetAngle; angle -= step) {
      servo.write(angle);
      currentAngle = angle;
      delay(delayMs);
    }
  }
  
  // Ensure final position is exactly the target
  servo.write(targetAngle);
  currentAngle = targetAngle;
}

// Function to move robot arm to home position
void moveToHome() {
  moveServoSmooth(servo5, servo5Angle, HOME_SERVO5); // Move gripper first
  moveServoSmooth(servo4, servo4Angle, HOME_SERVO4);
  moveServoSmooth(servo3, servo3Angle, HOME_SERVO3);
  moveServoSmooth(servo2, servo2Angle, HOME_SERVO2);
  moveServoSmooth(servo1, servo1Angle, HOME_SERVO1); // Move base last
}

// Function to send current servo positions back to Python
void sendFeedback() {
  Serial.print("Current angles:");
  Serial.print(servo1Angle);
  Serial.print(",");
  Serial.print(servo2Angle);
  Serial.print(",");
  Serial.print(servo3Angle);
  Serial.print(",");
  Serial.print(servo4Angle);
  Serial.print(",");
  Serial.println(servo5Angle);
}

void setup() {
  // Start serial communication
  Serial.begin(9600);
  
  // Attach servos to pins
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  servo4.attach(SERVO4_PIN);
  servo5.attach(SERVO5_PIN);
  
  // Optional: Setup for ultrasonic sensor
  // pinMode(TRIG_PIN, OUTPUT);
  // pinMode(ECHO_PIN, INPUT);
  
  // Initialize servos to home position
  servo1.write(HOME_SERVO1);
  servo2.write(HOME_SERVO2);
  servo3.write(HOME_SERVO3);
  servo4.write(HOME_SERVO4);
  servo5.write(HOME_SERVO5);
  
  // Set current angles to home positions
  servo1Angle = HOME_SERVO1;
  servo2Angle = HOME_SERVO2;
  servo3Angle = HOME_SERVO3;
  servo4Angle = HOME_SERVO4;
  servo5Angle = HOME_SERVO5;
  
  // Wait for servos to reach home position
  delay(1000);
  
  // Send initial feedback
  sendFeedback();
  
  Serial.println("Robot arm initialized and ready");
}

void loop() {
  // Read incoming commands from Python
  if (Serial.available() > 0) {
    // Read incoming string until newline
    String data = Serial.readStringUntil('\n');
    
    // Parse the comma-separated angles
    int values[5] = {-1, -1, -1, -1, -1}; // Initialize with invalid values
    int valueIndex = 0;
    int startPos = 0;
    int commaPos = data.indexOf(',');
    
    // Parse the CSV string
    while (commaPos >= 0 && valueIndex < 5) {
      values[valueIndex++] = data.substring(startPos, commaPos).toInt();
      startPos = commaPos + 1;
      commaPos = data.indexOf(',', startPos);
    }
    
    // Get the last value after the last comma
    if (startPos < data.length() && valueIndex < 5) {
      values[valueIndex++] = data.substring(startPos).toInt();
    }
    
    // Check if we got valid values for all 5 servos
    if (valueIndex == 5 && values[0] >= 0 && values[1] >= 0 && 
        values[2] >= 0 && values[3] >= 0 && values[4] >= 0) {
      
      // Constrain angles to valid range
      int targetAngles[5];
      for (int i = 0; i < 5; i++) {
        targetAngles[i] = constrain(values[i], 0, 180);
      }
      
      // Update servo positions
      servo1.write(targetAngles[0]);
      servo2.write(targetAngles[1]);
      servo3.write(targetAngles[2]);
      servo4.write(targetAngles[3]);
      servo5.write(targetAngles[4]);
      
      // Update current angle variables
      servo1Angle = targetAngles[0];
      servo2Angle = targetAngles[1];
      servo3Angle = targetAngles[2];
      servo4Angle = targetAngles[3];
      servo5Angle = targetAngles[4];
      
      // Send feedback
      sendFeedback();
    }
  }
  
  // Periodically send feedback to Python
  unsigned long currentTime = millis();
  if (currentTime - lastFeedbackTime >= FEEDBACK_INTERVAL) {
    sendFeedback();
    lastFeedbackTime = currentTime;
  }
  
  // Optional: Code for ultrasonic sensor reading
  /*
  // Clear the trigger pin
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  
  // Set the trigger pin high for 10 microseconds
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read the echo pin (duration in microseconds)
  long duration = pulseIn(ECHO_PIN, HIGH);
  
  // Calculate distance in centimeters
  float distance = duration * 0.034 / 2;
  
  // If object is detected within range, send information back to Python
  if (distance < 30) {
    Serial.print("Distance:");
    Serial.println(distance);
  }
  */
  
  // Small delay between loop iterations
  delay(15);
}
