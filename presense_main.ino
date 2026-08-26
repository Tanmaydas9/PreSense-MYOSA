/*
 * presense_main.ino
 * ==============================================================================
 * PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
 * Lead Developer: Tanmay Das
 * IEEE Sensors Council MYOSA Ideation and Prototype Competition
 * ------------------------------------------------------------------------------
 * Board   : MYOSA ESP32
 * Role    : Real-time edge inference loop. Samples IMU + FSR + barometer,
 *           calls predict_instability() (external, defined in inference.cpp /
 *           model_inference.h), and on a positive prediction fires a 200 ms
 *           non-blocking HIGH pulse on the NMES trigger pin.
 *
 * Peripherals
 *   I2C : MPU6050 (0x68), BMP180 (0x77), APDS9960 (0x39), SSD1306 OLED (0x3C)
 *   SPI : MicroSD card module (VSPI: SCK=18, MISO=19, MOSI=23, CS=5)
 *   ADC1: 4x FSR (BLE-safe; ADC2 is unusable while WiFi/BLE is active)
 *
 * Libraries required (Arduino Library Manager):
 *   - Adafruit MPU6050 + Adafruit Unified Sensor
 *   - Adafruit BMP085 Unified (covers BMP180)
 *   - Adafruit APDS9960
 *   - Adafruit SSD1306 + Adafruit GFX
 *   - SD (bundled with ESP32 core)
 * ==============================================================================
 */

#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_BMP085_U.h>
#include <Adafruit_APDS9960.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Inference entry point (implemented separately, e.g. inference.cpp using
// the TFLite Micro model exported as presense_motion_model.h).
#include "model_inference.h"   // bool predict_instability(const SensorWindow&);

// ===================== PIN MAP (raw int; 'D' label in comment) ==============
static constexpr int FSR_HEEL      = 32;   // D32 on board (ADC1_CH4)
static constexpr int FSR_TOE_L     = 33;   // D33 on board (ADC1_CH5)
static constexpr int FSR_TOE_R     = 34;   // D34 on board (ADC1_CH6, input-only)
static constexpr int FSR_MID       = 35;   // D35 on board (ADC1_CH7, input-only)
static constexpr int NMES_TRIGGER  = 14;   // D14 on board -> TENS/NMES + buzzer

static constexpr int I2C_SDA       = 21;   // D21 default ESP32 SDA
static constexpr int I2C_SCL       = 22;   // D22 default ESP32 SCL

static constexpr int SD_CS         = 5;    // D5  (VSPI CS)
static constexpr int SD_SCK        = 18;   // D18
static constexpr int SD_MISO       = 19;   // D19
static constexpr int SD_MOSI       = 23;   // D23

// ===================== TIMING ===============================================
static constexpr uint32_t SAMPLE_PERIOD_MS   = 10;    // 100 Hz sensor sampling
static constexpr uint32_t NMES_PULSE_MS      = 200;   // stimulation duration
static constexpr uint32_t NMES_REFRACTORY_MS = 800;   // min gap between triggers
static constexpr uint32_t OLED_REFRESH_MS    = 250;   // 4 Hz display update
static constexpr uint32_t SD_FLUSH_MS        = 2000;  // flush log every 2 s

// ===================== OBJECTS ==============================================
Adafruit_MPU6050  mpu;
Adafruit_BMP085_U bmp;
Adafruit_APDS9960 apds;
Adafruit_SSD1306  oled(128, 64, &Wire, -1);

// ===================== STATE ================================================
struct SensorFrame {
  float acc_x, acc_y, acc_z;
  float gyro_x, gyro_y, gyro_z;
  int   fsr_heel, fsr_toe_left, fsr_toe_right, fsr_mid;
  float altitude_delta;
};

static SensorFrame frame;

// Non-blocking NMES pulse state
static bool     nmesActive       = false;
static uint32_t nmesStartMs      = 0;
static uint32_t nmesLastFireMs   = 0;

// Scheduler timestamps
static uint32_t lastSampleMs     = 0;
static uint32_t lastOledMs       = 0;
static uint32_t lastFlushMs      = 0;

// Altitude baseline (rolling average of BMP180)
static float altitudeBaseline    = 0.0f;
static bool  baselinePrimed      = false;

// SD logging
static File   logFile;
static bool   sdOk               = false;
static uint32_t eventCount       = 0;

// Latest inference result (for OLED display)
static bool   lastPrediction     = false;
static char   lastGestureMsg[24] = "none";

// ===================== SETUP HELPERS ========================================
bool initMPU6050() {
  if (!mpu.begin(0x68, &Wire)) return false;
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);     // matches synthetic data (g)
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);          // matches synthetic data (dps)
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);       // anti-alias for 100 Hz loop
  return true;
}

bool initBMP180() {
  return bmp.begin(BMP085_MODE_STANDARD, &Wire);
}

bool initAPDS9960() {
  if (!apds.begin(10, APDS9960_AGAIN_4X, 0x39, &Wire)) return false;
  apds.enableProximity(true);
  apds.enableGesture(true);
  return true;
}

bool initOLED() {
  if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) return false;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);
  oled.println(F("PreSense v1.0"));
  oled.println(F("Init OK"));
  oled.display();
  return true;
}

bool initSD() {
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS)) return false;

  logFile = SD.open("/presense_log.csv", FILE_APPEND);
  if (!logFile) return false;

  // Write header only if the file is fresh
  if (logFile.size() == 0) {
    logFile.println(F("timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,"
                      "fsr_heel,fsr_toe_left,fsr_toe_right,fsr_mid,"
                      "altitude_delta,label"));
  }
  return true;
}

void fireNMES() {
  // Non-blocking start of the 200 ms stimulation pulse
  digitalWrite(NMES_TRIGGER, HIGH);
  nmesActive     = true;
  nmesStartMs    = millis();
  nmesLastFireMs = nmesStartMs;
  eventCount++;

  Serial.printf("[NMES] TRIGGER fired @ %lu ms (event #%lu)\n",
                (unsigned long)nmesStartMs, (unsigned long)eventCount);

  if (sdOk && logFile) {
    logFile.printf("%lu,EVENT,NMES_TRIGGER,%lu\n",
                   (unsigned long)nmesStartMs, (unsigned long)eventCount);
  }
}

void serviceNMES() {
  // Non-blocking pulse termination
  if (nmesActive && (millis() - nmesStartMs >= NMES_PULSE_MS)) {
    digitalWrite(NMES_TRIGGER, LOW);
    nmesActive = false;
    Serial.printf("[NMES] Pulse complete (%u ms), pin D14 LOW\n", NMES_PULSE_MS);
  }
}

// ===================== SENSOR READ ==========================================
void readSensors(SensorFrame& f) {
  // --- MPU6050 ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  f.acc_x  = a.acceleration.x / 9.80665f;   // -> g (match training units)
  f.acc_y  = a.acceleration.y / 9.80665f;
  f.acc_z  = a.acceleration.z / 9.80665f;
  f.gyro_x = g.gyro.x * 57.29578f;          // rad/s -> deg/s
  f.gyro_y = g.gyro.y * 57.29578f;
  f.gyro_z = g.gyro.z * 57.29578f;

  // --- FSRs on ADC1 (12-bit, 0-1023-equivalent scale for training parity) ---
  // NOTE: ESP32 ADC is 12-bit (0-4095); >>2 scales to 0-1023 to match the
  // normalization constants baked into the trained model.
  f.fsr_heel      = analogRead(FSR_HEEL)  >> 2;
  f.fsr_toe_left  = analogRead(FSR_TOE_L) >> 2;
  f.fsr_toe_right = analogRead(FSR_TOE_R) >> 2;
  f.fsr_mid       = analogRead(FSR_MID)   >> 2;

  // --- BMP180 altitude delta ---
  float alt = bmp.readAltitude();   // meters, sea-level pressure default
  if (!baselinePrimed) {
    altitudeBaseline = alt;
    baselinePrimed   = true;
  }
  altitudeBaseline = 0.995f * altitudeBaseline + 0.005f * alt;  // slow EMA
  f.altitude_delta = alt - altitudeBaseline;
}

// ===================== GESTURE (APDS9960) ===================================
void pollGesture() {
  if (!apds.gestureValid()) return;
  uint8_t g = apds.readGesture();
  switch (g) {
    case APDS9960_UP:    strncpy(lastGestureMsg, "UP: calibrate", 24);    break;
    case APDS9960_DOWN:  strncpy(lastGestureMsg, "DOWN: baseline", 24);   break;
    case APDS9960_LEFT:  strncpy(lastGestureMsg, "LEFT: log mark", 24);   break;
    case APDS9960_RIGHT: strncpy(lastGestureMsg, "RIGHT: status", 24);    break;
    default: return;
  }
  Serial.printf("[GESTURE] %s\n", lastGestureMsg);

  // Simple gesture actions
  if (g == APDS9960_DOWN) {
    baselinePrimed = false;   // re-zero altitude baseline
    Serial.println(F("[CAL] Altitude baseline re-primed"));
  }
}

// ===================== LOGGING ==============================================
void logFrame(const SensorFrame& f, int label) {
  if (!sdOk || !logFile) return;
  logFile.printf("%lu,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.5f,%d\n",
                 (unsigned long)millis(),
                 f.acc_x, f.acc_y, f.acc_z,
                 f.gyro_x, f.gyro_y, f.gyro_z,
                 f.fsr_heel, f.fsr_toe_left, f.fsr_toe_right, f.fsr_mid,
                 f.altitude_delta, label);
}

// ===================== OLED =================================================
void refreshOLED() {
  oled.clearDisplay();
  oled.setCursor(0, 0);
  oled.println(F("PreSense  ACTIVE"));
  oled.print(F("Gait: "));
  oled.println(lastPrediction ? F("FOOT DROP!") : F("Normal"));
  oled.print(F("FSR H/T: "));
  oled.print(frame.fsr_heel);
  oled.print(F("/"));
  oled.println((frame.fsr_toe_left + frame.fsr_toe_right) / 2);
  oled.print(F("dAlt: ")); oled.print(frame.altitude_delta, 3); oled.println(F(" m"));
  oled.print(F("NMES events: ")); oled.println(eventCount);
  oled.print(F("Gesture: ")); oled.println(lastGestureMsg);
  oled.print(F("NMES: ")); oled.println(nmesActive ? F("FIRING") : F("idle"));
  oled.display();
}

// ===================== SETUP ================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println(F("\n[PreSense] Booting..."));

  // --- GPIO ---
  pinMode(NMES_TRIGGER, OUTPUT);
  digitalWrite(NMES_TRIGGER, LOW);    // never default to stimulated
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);     // full ~3.3 V range on ADC1

  // --- I2C bus ---
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);              // 400 kHz fast mode for 100 Hz loop

  // --- Peripherals (each with debug print) ---
  Serial.print(F("[INIT] MPU6050 ... "));
  Serial.println(initMPU6050() ? F("OK") : F("FAIL (check 0x68 wiring)"));

  Serial.print(F("[INIT] BMP180  ... "));
  Serial.println(initBMP180() ? F("OK") : F("FAIL (check 0x77 wiring)"));

  Serial.print(F("[INIT] APDS9960... "));
  Serial.println(initAPDS9960() ? F("OK") : F("FAIL (check 0x39 wiring)"));

  Serial.print(F("[INIT] OLED    ... "));
  Serial.println(initOLED() ? F("OK") : F("FAIL (check 0x3C wiring)"));

  Serial.print(F("[INIT] SD card ... "));
  sdOk = initSD();
  Serial.println(sdOk ? F("OK") : F("FAIL (card absent - logging disabled)"));

  // --- Inference engine ---
  Serial.print(F("[INIT] TFLite Micro model ... "));
  Serial.println(inference_init() ? F("OK") : F("FAIL (model/arena)"));

  Serial.println(F("[PreSense] Setup complete. Entering 100 Hz loop."));
}

// ===================== MAIN LOOP (non-blocking) =============================
void loop() {
  const uint32_t now = millis();

  // ---- NMES pulse servicing runs EVERY iteration (highest priority) -------
  serviceNMES();

  // ---- 100 Hz sensor sampling + inference ---------------------------------
  if (now - lastSampleMs >= SAMPLE_PERIOD_MS) {
    lastSampleMs = now;

    readSensors(frame);

    // Debug stream (throttled internally by predict-side if needed)
    Serial.printf("[SENS] a(%.2f,%.2f,%.2f) g(%.1f,%.1f,%.1f) "
                  "fsr(%d,%d,%d,%d) dAlt=%.3f\n",
                  frame.acc_x, frame.acc_y, frame.acc_z,
                  frame.gyro_x, frame.gyro_y, frame.gyro_z,
                  frame.fsr_heel, frame.fsr_toe_left,
                  frame.fsr_toe_right, frame.fsr_mid,
                  frame.altitude_delta);

    // Push frame into the sliding-window buffer and run inference.
    // predict_instability() internally maintains the 128-sample ring buffer,
    // normalizes features, and invokes the MicroInterpreter.
    bool instability = predict_instability(frame);
    lastPrediction   = instability;

    logFrame(frame, instability ? 1 : 0);

    if (instability) {
      // Refractory check: avoid re-firing during the same stride
      if (!nmesActive && (now - nmesLastFireMs >= NMES_REFRACTORY_MS)) {
        Serial.println(F("[PRED] Instability detected -> scheduling NMES "
                         "(t-100 ms pre-contact window)"));
        fireNMES();
      } else {
        Serial.println(F("[PRED] Instability detected but NMES refractory "
                         "- skipping"));
      }
    }
  }

  // ---- Gesture calibration (polled, non-blocking) --------------------------
  pollGesture();

  // ---- OLED at 4 Hz --------------------------------------------------------
  if (now - lastOledMs >= OLED_REFRESH_MS) {
    lastOledMs = now;
    refreshOLED();
  }

  // ---- SD flush at 0.5 Hz (keeps FAT writes off the inference path) -------
  if (sdOk && logFile && (now - lastFlushMs >= SD_FLUSH_MS)) {
    lastFlushMs = now;
    logFile.flush();
  }
}

// model_inference.h
#pragma once
struct SensorFrame;   // forward decl (defined in presense_main.ino)
bool inference_init();
bool predict_instability(const SensorFrame& f);
