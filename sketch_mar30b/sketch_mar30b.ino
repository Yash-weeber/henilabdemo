#include <Servo.h>

Servo myServo; // Create a Servo object

void setup() {
  myServo.attach(6); // Attach servo to pin 9
  myServo.write(0); // Start at middle position (90 degrees)
  Serial.begin(9600); // Start serial communication at baud rate of 9600
}

void loop() {
  if (Serial.available() > 0) {
    String angleString = Serial.readStringUntil('\n'); // Read angle as string until newline character
    int angle = angleString.toInt(); // Convert string to integer

    // Constrain angle to valid range for servo (0-180 degrees)
    angle = constrain(angle, 0, 270);

    // Move servo to specified angle
    myServo.write(angle);
    
    delay(15); // Allow time for servo to move to position
  }
}
