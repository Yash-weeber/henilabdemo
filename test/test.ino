#include <Servo.h> // Include the Servo library

Servo myServo; // Create a servo object

void setup() {
  myServo.attach(6); // Attach the servo to pin 9
}

void loop() {
  // Move servo to 0 degrees
  myServo.write(0);
  delay(1000); // Wait for 1 second

  // Move servo to 90 degrees
  myServo.write(90);
  delay(1000); // Wait for 1 second

  // Move servo to 180 degrees
  myServo.write(180);
  delay(1000); // Wait for 1 second
}
