"""
motion_model_training.py
==============================================================================
PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
Lead Developer: Tanmay Das
IEEE Sensors Council MYOSA Ideation and Prototype Competition
------------------------------------------------------------------------------
Loads the CSV emitted by generate_synthetic_dataset.py, builds sliding-window
(time-series) samples, trains a lightweight 1D CNN (default) or LSTM that can
run inside TFLite for Microcontrollers on the MYOSA ESP32, and exports:

    presense_motion_model.tflite     - INT8 quantized flatbuffer
    presense_motion_model.h          - C byte-array header (constexpr unsigned char[])

Pipeline:
    CSV -> per-cycle normalization -> sliding windows -> train/val/test split
        -> model train -> INT8 post-training quantization (repr. dataset)
        -> .tflite + .h export -> latency / size report

Constraints honored for TFLM deployment on ESP32:
    * Only ops supported by TFLM are used (Conv1D, Pool, Dense, Softmax).
    * Full-integer quantization -> uint8 activations/weights.
    * Small tensor arena (window len 128 x 9 features) fits in <64 KB SRAM.
==============================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    csv_path: str = "presense_synthetic_gait.csv"
    window_size: int = 128           # 1.28 s @ 100 Hz - covers pre-swing to heel strike
    step: int = 16                   # window slide (samples)
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    model_type: str = "cnn"          # "cnn" (fast, recommended) or "lstm"
    out_tflite: str = "presense_motion_model.tflite"
    out_header: str = "presense_motion_model.h"
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

# Channels fed into the model (must match the ESP32 feature-pack order).
FEATURES = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "fsr_heel", "fsr_toe_left", "fsr_toe_right", "fsr_mid",
    "altitude_delta",
]
N_FEATURES = len(FEATURES)
CLASS_NAMES = ["Normal Gait", "Hesitation/Foot Drop"]


# ------------------------------------------------------------------
# Data loading & windowing
# ------------------------------------------------------------------
def load_and_window(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) as float32/int32."""
    df = pd.read_csv(cfg.csv_path)

    # --- Normalize per-channel with simple min-max, learned on TRAIN only ---
    # Split first by cycle_id so windows from one cycle never cross folds.
    groups = df["cycle_id"].values
    labels_full = df["label"].values

    gss_train = GroupShuffleSplit(n_splits=1, test_size=cfg.val_split + cfg.test_split,
                                  random_state=cfg.seed)
    train_idx, holdout_idx = next(gss_train.split(df, labels_full, groups))

    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    holdout_groups = holdout_df["cycle_id"].values
    holdout_labels = holdout_df["label"].values

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.test_split / (cfg.val_split + cfg.test_split),
        random_state=cfg.seed + 1,
    )
    val_idx, test_idx = next(gss_val.split(holdout_df, holdout_labels, holdout_groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)

    # --- Min-max per feature, fitted on train ---
    feat_min = train_df[FEATURES].min()
    feat_max = train_df[FEATURES].max()
    denom = (feat_max - feat_min).replace(0, 1.0)

    def norm(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d[FEATURES] = (d[FEATURES] - feat_min) / denom
        return d

    train_df = norm(train_df)
    val_df = norm(val_df)
    test_df = norm(test_df)

    def windows(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        # Window inside each cycle to avoid phase discontinuity at boundaries.
        for _, cycle_df in d.groupby("cycle_id"):
            arr = cycle_df[FEATURES].values.astype(np.float32)
            lab = int(cycle_df["label"].iloc[0])
            for start in range(0, len(arr) - cfg.window_size + 1, cfg.step):
                X.append(arr[start:start + cfg.window_size])
                y.append(lab)
        return np.stack(X), np.asarray(y, dtype=np.int32)

    X_train, y_train = windows(train_df)
    X_val, y_val = windows(val_df)
    X_test, y_test = windows(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
def build_cnn(cfg: TrainConfig) -> tf.keras.Model:
    """Compact 1D CNN, sized to run inside a ~64 KB tensor arena on ESP32."""
    inputs = tf.keras.Input(shape=(cfg.window_size, N_FEATURES), name="input")

    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)                       # -> 64
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)                       # -> 32
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)              # -> (32,)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="presense_cnn")
    return model


def build_lstm(cfg: TrainConfig) -> tf.keras.Model:
    """LSTM fallback; larger & slower than the CNN but kept as an option."""
    inputs = tf.keras.Input(shape=(cfg.window_size, N_FEATURES), name="input")
    x = tf.keras.layers.LSTM(24, return_sequences=False)(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="presense_lstm")


def build_model(cfg: TrainConfig) -> tf.keras.Model:
    return build_lstm(cfg) if cfg.model_type == "lstm" else build_cnn(cfg)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train(model: tf.keras.Model,
          X_train, y_train, X_val, y_val, cfg: TrainConfig) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-5),
    ]
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=cfg.batch_size,
              epochs=cfg.epochs,
              verbose=2,
              callbacks=callbacks,
              class_weight={0: 1.0, 1: 1.4})  # slight bias toward catching neuropathy


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(model: tf.keras.Model, X_test, y_test) -> None:
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n[PreSense] Test-set classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print("[PreSense] Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------------
# TFLite conversion (INT8 full-integer quantization)
# ------------------------------------------------------------------
def convert_to_tflite_int8(model: tf.keras.Model,
                           X_train: np.ndarray,
                           cfg: TrainConfig) -> bytes:
    """Return the serialized INT8 TFLite flatbuffer."""

    def representative_dataset():
        # Small representative subset for activation range calibration.
        for i in range(0, len(X_train), max(1, len(X_train) // 200)):
            yield [X_train[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


# ------------------------------------------------------------------
# C header export
# ------------------------------------------------------------------
def write_c_header(tflite_bytes: bytes, out_path: Path,
                   array_name: str = "presense_motion_model_tflite") -> None:
    """Emit a C header that the ESP32 firmware can #include directly."""
    lines = [
        "// Auto-generated by motion_model_training.py",
        "// PreSense - Predictive Sensory Augmentation Platform",
        "// Model: INT8 quantized TFLite flatbuffer for TFLite Micro",
        "#ifndef PRESENSE_MOTION_MODEL_H",
        "#define PRESENSE_MOTION_MODEL_H",
        "",
        "#include <cstdint>",
        "",
        f"constexpr unsigned int {array_name}_len = {len(tflite_bytes)};",
        f"alignas(8) const unsigned char {array_name}[] = {{",
    ]
    for i in range(0, len(tflite_bytes), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 12])
        lines.append(f"    {chunk},")
    lines.extend(["};", "", "#endif  // PRESENSE_MOTION_MODEL_H", ""])
    out_path.write_text("\n".join(lines))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + export PreSense motion model.")
    p.add_argument("--csv", default=TrainConfig.csv_path,
                   help="Path to CSV from generate_synthetic_dataset.py.")
    p.add_argument("--window", type=int, default=TrainConfig.window_size,
                   help="Sliding window length in samples (default 128).")
    p.add_argument("--step", type=int, default=TrainConfig.step,
                   help="Window stride (default 16).")
    p.add_argument("--model", choices=["cnn", "lstm"], default=TrainConfig.model_type,
                   help="Model architecture (default cnn).")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch", type=int, default=TrainConfig.batch_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--out-tflite", default=TrainConfig.out_tflite)
    p.add_argument("--out-header", default=TrainConfig.out_header)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        csv_path=args.csv,
        window_size=args.window,
        step=args.step,
        batch_size=args.batch,
        epochs=args.epochs,
        model_type=args.model,
        out_tflite=args.out_tflite,
        out_header=args.out_header,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    print(f"[PreSense] Loading {cfg.csv_path} ...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_window(cfg)
    print(f"[PreSense] Windows: train={X_train.shape}, val={X_val.shape}, "
          f"test={X_test.shape}")

    model = build_model(cfg)
    model.summary()

    print(f"[PreSense] Training {cfg.model_type.upper()} for up to {cfg.epochs} epochs ...")
    train(model, X_train, y_train, X_val, y_val, cfg)

    evaluate(model, X_test, y_test)

    # ---------------- Convert to INT8 TFLite -----------------------
    print("[PreSense] Converting to INT8 TFLite ...")
    tflite_bytes = convert_to_tflite_int8(model, X_train, cfg)

    tflite_path = Path(cfg.out_tflite).resolve()
    tflite_path.write_bytes(tflite_bytes)
    print(f"[PreSense] Wrote {tflite_path.name} ({len(tflite_bytes)/1024:.1f} KB)")

    header_path = Path(cfg.out_header).resolve()
    write_c_header(tflite_bytes, header_path)
    print(f"[PreSense] Wrote {header_path.name}")

    # ---------------- Sanity check with TFLite interpreter ---------
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    print(f"[PreSense] TFLite input  : {in_details['shape']} dtype={in_details['dtype']}")
    print(f"[PreSense] TFLite output : {out_details['shape']} dtype={out_details['dtype']}")

    # Quick smoke test on one window
    sample = X_test[:1]
    scale, zero_point = in_details["quantization"]
    q_sample = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)
    interpreter.set_tensor(in_details["index"], q_sample)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_details["index"])
    print(f"[PreSense] Quantized inference on sample -> {q_out}")

    print("\n[PreSense] Done. Flash presense_motion_model.h into your ESP32 "
          "firmware and feed it to tflite::MicroInterpreter.")


if __name__ == "__main__":
    main()

#python motion_model_training.py \
 #   --csv presense_synthetic_gait.csv \
 #   --model cnn \
  #  --window 128 --step 16 \
 #   --epochs 40 --batch 64
print("[PreSense] Paste into presense_inference.cpp FEAT_MIN / FEAT_MAX:")
print("FEAT_MIN: " + ", ".join(f"{feat_min[n]:.6f}f" for n in FEATURES))
print("FEAT_MAX: " + ", ".join(f"{feat_max[n]:.6f}f" for n in FEATURES))
