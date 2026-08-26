/*
 * firmware_v3_ml_extension.cpp
 * =============================================================================
 * PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
 * Lead Developer: Tanmay Das
 * IEEE Sensors Council MYOSA Ideation and Prototype Competition
 * -----------------------------------------------------------------------------
 * Lightweight instability-prediction extension for the MYOSA ESP32 firmware.
 *
 * This file implements:
 *
 *     bool predict_instability(float* sensor_data)
 *
 * Expected sensor_data layout, 11 float values:
 *
 *     [0]  acc_x              g
 *     [1]  acc_y              g
 *     [2]  acc_z              g
 *     [3]  gyro_x             deg/s
 *     [4]  gyro_y             deg/s
 *     [5]  gyro_z             deg/s
 *     [6]  fsr_heel           0..1023
 *     [7]  fsr_toe_left       0..1023
 *     [8]  fsr_toe_right      0..1023
 *     [9]  fsr_mid            0..1023
 *     [10] altitude_delta     meters relative to BMP180 baseline
 *
 * The current implementation is a real-time heuristic intended for early
 * firmware integration. It is stateful and should be called once per sensor
 * sample, normally every 10 ms from the 100 Hz loop.
 *
 * A disabled TensorFlow Lite Micro placeholder is included near the bottom.
 * =============================================================================
 */

#include <Arduino.h>
#include <math.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Build options
// -----------------------------------------------------------------------------

#ifndef PRESENSE_DEBUG_ML
#define PRESENSE_DEBUG_ML 1
#endif

// Keep this at 0 until the INT8 TFLite Micro model has been validated against
// logged ESP32 data. The heuristic remains active while this is 0.
#ifndef PRESENSE_USE_TFLITE
#define PRESENSE_USE_TFLITE 0
#endif

// -----------------------------------------------------------------------------
// Sensor vector indices
// -----------------------------------------------------------------------------

enum PresenseFeatureIndex : uint8_t {
  IDX_ACC_X = 0,
  IDX_ACC_Y = 1,
  IDX_ACC_Z = 2,
  IDX_GYRO_X = 3,
  IDX_GYRO_Y = 4,
  IDX_GYRO_Z = 5,
  IDX_FSR_HEEL = 6,
  IDX_FSR_TOE_LEFT = 7,
  IDX_FSR_TOE_RIGHT = 8,
  IDX_FSR_MID = 9,
  IDX_ALTITUDE_DELTA = 10
};

static constexpr uint8_t PRESENSE_FEATURE_COUNT = 11;

// -----------------------------------------------------------------------------
// Heuristic thresholds
//
// Units must match the values prepared by presense_main.ino:
//   acceleration : g
//   gyroscope    : deg/s
//   FSR          : 0..1023 ADC-equivalent counts
//   altitude     : meters above/below rolling baseline
// -----------------------------------------------------------------------------

static constexpr float HEEL_CONTACT_THRESHOLD       = 260.0f;
static constexpr float TOE_LOADED_THRESHOLD         = 210.0f;
static constexpr float MID_LOADED_THRESHOLD         = 180.0f;

// If the heel is loaded but the toes have not loaded after this interval,
// treat it as delayed plantar-pressure transfer / hesitation.
static constexpr uint32_t DELAYED_PRESSURE_MS       = 180;

// Low foot-clearance and weak swing/dorsiflexion thresholds.
static constexpr float LOW_CLEARANCE_M              = 0.025f;
static constexpr float WEAK_SWING_GYRO_Y_DPS        = 55.0f;
static constexpr float WEAK_FORWARD_ACC_G           = 0.22f;

// Mediolateral instability and abnormal roll/yaw correction.
static constexpr float WOBBLE_GYRO_X_DPS            = 135.0f;
static constexpr float WOBBLE_GYRO_Z_DPS            = 150.0f;

// Excessive difference between medial and lateral toe loading.
static constexpr float TOE_ASYMMETRY_COUNTS         = 230.0f;

// Fusion decision thresholds with hysteresis.
static constexpr float INSTABILITY_ENTER_SCORE      = 0.62f;
static constexpr float INSTABILITY_EXIT_SCORE       = 0.45f;

// Smoothing constants.
static constexpr float SCORE_EMA_ALPHA              = 0.30f;
static constexpr float SIGNAL_EMA_ALPHA             = 0.25f;

// -----------------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------------

struct HeuristicState {
  uint32_t lastUpdateMs;

  // Heel-contact timing.
  bool heelContactActive;
  uint32_t heelContactStartMs;

  // Smooth sensor channels.
  float emaHeel;
  float emaToeAverage;
  float emaMid;
  float emaAbsGyroY;
  float emaAbsGyroZ;
  float emaAltitudeDelta;

  // Score and latched output.
  float scoreEma;
  bool instabilityLatched;

  // Diagnostics.
  float delayedPressureScore;
  float weakSwingScore;
  float lowClearanceScore;
  float wobbleScore;
  float asymmetryScore;
};

static HeuristicState g_ml = {
  0,        // lastUpdateMs
  false,    // heelContactActive
  0,        // heelContactStartMs
  0.0f,     // emaHeel
  0.0f,     // emaToeAverage
  0.0f,     // emaMid
  0.0f,     // emaAbsGyroY
  0.0f,     // emaAbsGyroZ
  0.0f,     // emaAltitudeDelta
  0.0f,     // scoreEma
  false,    // instabilityLatched
  0.0f,     // delayedPressureScore
  0.0f,     // weakSwingScore
  0.0f,     // lowClearanceScore
  0.0f,     // wobbleScore
  0.0f      // asymmetryScore
};

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

static inline float clamp01(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 1.0f) {
    return 1.0f;
  }
  return value;
}

static inline float emaUpdate(float previous, float sample, float alpha) {
  return previous + alpha * (sample - previous);
}

static bool sensorVectorIsValid(const float* sensorData) {
  if (sensorData == nullptr) {
    return false;
  }

  for (uint8_t i = 0; i < PRESENSE_FEATURE_COUNT; ++i) {
    if (!isfinite(sensorData[i])) {
      return false;
    }
  }

  return true;
}

// -----------------------------------------------------------------------------
// Feature calculations
// -----------------------------------------------------------------------------

static float calculateDelayedPressureScore(uint32_t nowMs,
                                           float heel,
                                           float toeAverage,
                                           float mid) {
  const bool heelLoaded = heel >= HEEL_CONTACT_THRESHOLD;

  if (!heelLoaded) {
    g_ml.heelContactActive = false;
    g_ml.heelContactStartMs = nowMs;
    return 0.0f;
  }

  if (!g_ml.heelContactActive) {
    g_ml.heelContactActive = true;
    g_ml.heelContactStartMs = nowMs;
    return 0.0f;
  }

  // A normal step should transfer pressure toward midfoot and forefoot soon
  // after heel contact. Persistent heel-only loading indicates hesitation.
  const bool forefootLoaded =
      (toeAverage >= TOE_LOADED_THRESHOLD) ||
      (mid >= MID_LOADED_THRESHOLD);

  if (forefootLoaded) {
    return 0.0f;
  }

  const uint32_t heelOnlyDuration = nowMs - g_ml.heelContactStartMs;

  if (heelOnlyDuration <= DELAYED_PRESSURE_MS) {
    return 0.0f;
  }

  // Grow the score for up to another 220 ms after the threshold.
  const float excessMs =
      static_cast<float>(heelOnlyDuration - DELAYED_PRESSURE_MS);

  return clamp01(excessMs / 220.0f);
}

static float calculateWeakSwingScore(float accX,
                                     float absGyroY,
                                     float altitudeDelta) {
  const float weakAngularComponent =
      clamp01((WEAK_SWING_GYRO_Y_DPS - absGyroY) /
              WEAK_SWING_GYRO_Y_DPS);

  const float weakForwardComponent =
      clamp01((WEAK_FORWARD_ACC_G - fabsf(accX)) /
              WEAK_FORWARD_ACC_G);

  const float lowClearanceComponent =
      clamp01((LOW_CLEARANCE_M - altitudeDelta) /
              LOW_CLEARANCE_M);

  return clamp01(
      0.40f * weakAngularComponent +
      0.25f * weakForwardComponent +
      0.35f * lowClearanceComponent
  );
}

static float calculateWobbleScore(float gyroX, float gyroZ) {
  const float rollComponent =
      clamp01(fabsf(gyroX) / WOBBLE_GYRO_X_DPS);

  const float yawComponent =
      clamp01(fabsf(gyroZ) / WOBBLE_GYRO_Z_DPS);

  return clamp01(0.45f * rollComponent + 0.55f * yawComponent);
}

static float calculateToeAsymmetryScore(float toeLeft, float toeRight) {
  const float difference = fabsf(toeLeft - toeRight);

  if (difference <= TOE_ASYMMETRY_COUNTS) {
    return 0.0f;
  }

  return clamp01((difference - TOE_ASYMMETRY_COUNTS) /
                 TOE_ASYMMETRY_COUNTS);
}

// -----------------------------------------------------------------------------
// Public heuristic API
// -----------------------------------------------------------------------------

bool predict_instability(float* sensor_data) {
  if (!sensorVectorIsValid(sensor_data)) {
#if PRESENSE_DEBUG_ML
    Serial.println(F("[ML] Invalid sensor vector; holding last decision"));
#endif
    return g_ml.instabilityLatched;
  }

  const uint32_t nowMs = millis();

  if (g_ml.lastUpdateMs == 0) {
    g_ml.lastUpdateMs = nowMs;
  }

  // Extract features.
  const float accX          = sensor_data[IDX_ACC_X];
  const float gyroX         = sensor_data[IDX_GYRO_X];
  const float gyroY         = sensor_data[IDX_GYRO_Y];
  const float gyroZ         = sensor_data[IDX_GYRO_Z];
  const float heel          = sensor_data[IDX_FSR_HEEL];
  const float toeLeft       = sensor_data[IDX_FSR_TOE_LEFT];
  const float toeRight      = sensor_data[IDX_FSR_TOE_RIGHT];
  const float mid           = sensor_data[IDX_FSR_MID];
  const float altitudeDelta = sensor_data[IDX_ALTITUDE_DELTA];

  const float toeAverage = 0.5f * (toeLeft + toeRight);

  // Smooth the noisy pressure and motion channels.
  g_ml.emaHeel =
      emaUpdate(g_ml.emaHeel, heel, SIGNAL_EMA_ALPHA);

  g_ml.emaToeAverage =
      emaUpdate(g_ml.emaToeAverage, toeAverage, SIGNAL_EMA_ALPHA);

  g_ml.emaMid =
      emaUpdate(g_ml.emaMid, mid, SIGNAL_EMA_ALPHA);

  g_ml.emaAbsGyroY =
      emaUpdate(g_ml.emaAbsGyroY, fabsf(gyroY), SIGNAL_EMA_ALPHA);

  g_ml.emaAbsGyroZ =
      emaUpdate(g_ml.emaAbsGyroZ, fabsf(gyroZ), SIGNAL_EMA_ALPHA);

  g_ml.emaAltitudeDelta =
      emaUpdate(g_ml.emaAltitudeDelta, altitudeDelta, SIGNAL_EMA_ALPHA);

  // Calculate individual neuropathy indicators.
  g_ml.delayedPressureScore =
      calculateDelayedPressureScore(
          nowMs,
          g_ml.emaHeel,
          g_ml.emaToeAverage,
          g_ml.emaMid
      );

  g_ml.weakSwingScore =
      calculateWeakSwingScore(
          accX,
          g_ml.emaAbsGyroY,
          g_ml.emaAltitudeDelta
      );

  g_ml.lowClearanceScore =
      clamp01((LOW_CLEARANCE_M - g_ml.emaAltitudeDelta) /
              LOW_CLEARANCE_M);

  g_ml.wobbleScore =
      calculateWobbleScore(gyroX, gyroZ);

  g_ml.asymmetryScore =
      calculateToeAsymmetryScore(toeLeft, toeRight);

  // Weighted late fusion. Delayed pressure buildup and weak swing receive the
  // largest weights because they correspond most directly to the anticipated
  // foot-drop intervention.
  const float rawScore =
      0.36f * g_ml.delayedPressureScore +
      0.27f * g_ml.weakSwingScore +
      0.14f * g_ml.lowClearanceScore +
      0.15f * g_ml.wobbleScore +
      0.08f * g_ml.asymmetryScore;

  g_ml.scoreEma =
      emaUpdate(g_ml.scoreEma, rawScore, SCORE_EMA_ALPHA);

  // Hysteresis avoids rapid flickering between true and false.
  if (!g_ml.instabilityLatched &&
      g_ml.scoreEma >= INSTABILITY_ENTER_SCORE) {
    g_ml.instabilityLatched = true;

#if PRESENSE_DEBUG_ML
    Serial.printf(
      "[ML] INSTABILITY ENTER score=%.3f | delay=%.2f weak=%.2f "
      "clear=%.2f wobble=%.2f asym=%.2f\n",
      g_ml.scoreEma,
      g_ml.delayedPressureScore,
      g_ml.weakSwingScore,
      g_ml.lowClearanceScore,
      g_ml.wobbleScore,
      g_ml.asymmetryScore
    );
#endif
  } else if (g_ml.instabilityLatched &&
             g_ml.scoreEma <= INSTABILITY_EXIT_SCORE) {
    g_ml.instabilityLatched = false;

#if PRESENSE_DEBUG_ML
    Serial.printf(
      "[ML] Instability exit score=%.3f\n",
      g_ml.scoreEma
    );
#endif
  }

  g_ml.lastUpdateMs = nowMs;
  return g_ml.instabilityLatched;
}

// Optional diagnostics for OLED, BLE telemetry, or SD-card logging.
float presense_ml_instability_score() {
  return g_ml.scoreEma;
}

float presense_ml_delayed_pressure_score() {
  return g_ml.delayedPressureScore;
}

float presense_ml_weak_swing_score() {
  return g_ml.weakSwingScore;
}

void presense_ml_reset() {
  g_ml.lastUpdateMs = 0;
  g_ml.heelContactActive = false;
  g_ml.heelContactStartMs = 0;

  g_ml.emaHeel = 0.0f;
  g_ml.emaToeAverage = 0.0f;
  g_ml.emaMid = 0.0f;
  g_ml.emaAbsGyroY = 0.0f;
  g_ml.emaAbsGyroZ = 0.0f;
  g_ml.emaAltitudeDelta = 0.0f;

  g_ml.scoreEma = 0.0f;
  g_ml.instabilityLatched = false;

  g_ml.delayedPressureScore = 0.0f;
  g_ml.weakSwingScore = 0.0f;
  g_ml.lowClearanceScore = 0.0f;
  g_ml.wobbleScore = 0.0f;
  g_ml.asymmetryScore = 0.0f;
}

// -----------------------------------------------------------------------------
// TensorFlow Lite Micro placeholder
//
// This block is intentionally disabled while PRESENSE_USE_TFLITE is 0. Once
// presense_motion_model.h and the TFLite Micro runtime are added, replace the
// heuristic decision above with the quantized-model output probability.
// -----------------------------------------------------------------------------

#if PRESENSE_USE_TFLITE

#include "presense_motion_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr uint32_t kTensorArenaSize = 48 * 1024;
alignas(16) uint8_t g_tensorArena[kTensorArenaSize];

tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
bool g_tfliteReady = false;

}  // namespace

bool presense_tflite_init() {
  tflite::InitializeTarget();

  const tflite::Model* model =
      tflite::GetModel(presense_motion_model_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[TFLM] Model schema version mismatch"));
    return false;
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddRelu();
  resolver.AddQuantize();

  static tflite::MicroInterpreter interpreter(
      model,
      resolver,
      g_tensorArena,
      kTensorArenaSize
  );

  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[TFLM] AllocateTensors failed"));
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  g_tfliteReady = true;

  Serial.printf(
      "[TFLM] Ready. Arena used: %u bytes\n",
      static_cast<unsigned>(g_interpreter->arena_used_bytes())
  );

  return true;
}

bool presense_tflite_ready() {
  return g_tfliteReady;
}

#endif  // PRESENSE_USE_TFLITE
