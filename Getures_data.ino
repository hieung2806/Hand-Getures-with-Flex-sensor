

#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
BluetoothSerial SerialBT;
const int potPin = 36;
const int potPin1=39;
const int potPin2 =34;
const int potPin3=35;
const int potPin4=32;
float accX;
float accY;
float accZ;
int pinky;
int ring;
int indexfinger;
int thumb;
int middle;
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("HAND GESTURE RECOGNITION"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");
  if (!mpu.begin()) {
    SerialBT.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  SerialBT.println("");
  delay(100);
  }

  
void loop() {
  pinky = analogRead(potPin);
  ring= analogRead(potPin1);
  indexfinger= analogRead(potPin2);
  thumb= analogRead(potPin3);
  middle= analogRead(potPin4);
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  accX = a.acceleration.x;
  accY = a.acceleration.y;
  accZ = a.acceleration.z;


  //if (SerialBT.available()) {
  SerialBT.print(accX);
  SerialBT.print(",");
  SerialBT.print(accY);
  SerialBT.print(",");
  SerialBT.print(accZ);
  SerialBT.print(",");
  SerialBT.print(pinky);
  SerialBT.print(",");
  SerialBT.print(ring);
  SerialBT.print(",");
  SerialBT.print(middle);
  SerialBT.print(",");
  SerialBT.print(indexfinger);
  SerialBT.print(",");
  SerialBT.print(thumb);
  SerialBT.println("");
  //}
}
