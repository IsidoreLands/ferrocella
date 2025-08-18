#define HELIX1_PIN 9
#define HELIX2_PIN 10
#define HALL_PIN A0
#define LCR_TX 8
#define LCR_RX 7
#define LASER_PIN 11
#define PHOTO_PIN A1
#include <SoftwareSerial.h>
SoftwareSerial lcrSerial(LCR_RX, LCR_TX);

void setup() {
  pinMode(HELIX1_PIN, OUTPUT);
  pinMode(HELIX2_PIN, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);
  pinMode(HALL_PIN, INPUT);
  pinMode(PHOTO_PIN, INPUT);
  lcrSerial.begin(9600);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'D') { analogWrite(HELIX1_PIN, 255); analogWrite(HELIX2_PIN, 0); }
    else if (cmd == 'A') { analogWrite(HELIX1_PIN, 128); analogWrite(HELIX2_PIN, 128); }
    else if (cmd == 'O') { analogWrite(HELIX1_PIN, 0); analogWrite(HELIX2_PIN, 0); }
    else if (cmd == 'L') { int pulse = Serial.parseInt(); analogWrite(LASER_PIN, pulse); }
  }
  int hallValue = analogRead(HALL_PIN);
  float magneticField = (hallValue - 512) * 0.00488;
  int photoValue = analogRead(PHOTO_PIN);
  float laserIntensity = photoValue * 0.00488;
  lcrSerial.println("L?");
  delay(100);
  String lcrData = lcrSerial.readStringUntil('\n');
  Serial.print("HALL:"); Serial.print(magneticField);
  Serial.print(",LCR:"); Serial.print(lcrData);
  Serial.print(",LASER:"); Serial.println(laserIntensity);
  delay(500);
}
