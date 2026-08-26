/*
 * presense_inference.h
 * ==============================================================================
 * PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
 * Lead Developer: Tanmay Das
 * IEEE Sensors Council MYOSA Ideation and Prototype Competition
 * ------------------------------------------------------------------------------
 * TensorFlow Lite Micro inference wrapper for the MYOSA ESP32.
 *
 * Responsibilities:
 *   - Load the INT8 quantized model (presense_motion_model.h)
 *   - Maintain a 128-sample sliding ring buffer of 11-channel sensor frames
 *   - Apply the exact min-max normalization fitted during training
 *   - Quantize the input window to int8 (scale / zero_point from the model)
 *   - Run the MicroInterpreter and expose predict_instability()
 *
 * Usage (from presense_main.ino):
 *   inference_init();                        // once, in setup()
 *   bool hit = predict_instability(frame);   // every 10 ms sample tick
 * ==============================================================================
 */

#ifndef PRESENSE_INFERENCE_H
#define PRESENSE_INFERENCE_H

#include <cstdint>

// Must match the SensorFrame struct defined in presense_main.ino.
// Kept as an identical standalone declaration so this module compiles
// independently while the layout stays shared.
struct SensorFrame {
  float acc_x, acc_y, acc_z;
  float gyro_x, gyro_y, gyro_z;
  int   fsr_heel, fsr_toe_left, fsr_toe_right, fsr_mid;
  float altitude_delta;
};

// ---- Lifecycle --------------------------------------------------------------
// Allocates the tensor arena, builds the interpreter, verifies model schema
// and input geometry. Returns true on success. Call once from setup().
bool inference_init();

// ---- Inference --------------------------------------------------------------
// Push one SensorFrame into the ring buffer. Once the buffer holds a full
// 128-sample window, run INT8 inference (every INFERENCE_STRIDE ticks) and
// return the instability decision. Returns false during warmup.
bool predict_instability(const SensorFrame& f);

// ---- Diagnostics ------------------------------------------------------------
// Last dequantized softmax probability for class 1 (foot drop), in [0, 1].
float    inference_last_probability();

// Measured MicroInterpreter invoke time of the last run, in microseconds.
uint32_t inference_last_latency_us();

// Number of samples currently in the ring buffer (0..128).
uint16_t inference_buffer_fill();

#endif  // PRESENSE_INFERENCE_H
