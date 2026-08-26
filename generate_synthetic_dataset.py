"""
generate_synthetic_dataset.py
==============================================================================
PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
Lead Developer: Tanmay Das
IEEE Sensors Council MYOSA Ideation and Prototype Competition
------------------------------------------------------------------------------
Synthesizes a CSV dataset of 500 gait cycles that mirrors what the MYOSA ESP32
would stream from:
    - MPU6050  : acc_x/y/z, gyro_x/y/z
    - 4x FSR   : fsr_heel, fsr_toe_left, fsr_toe_right, fsr_mid
    - BMP180   : altitude_delta (relative to a rolling baseline)

Label semantics
    0 -> Normal gait
    1 -> Hesitation / foot drop (neuropathy-affected cycle)

Neuropathy corruption applied to label==1 cycles:
    * Delayed and blunted plantar pressure buildup (FSR heel/toe onset shifted
      later in the cycle, peak amplitude attenuated, ramp slope reduced).
    * Reduced dorsiflexion during swing -> smaller acc_x/gyro_y toe-off peak.
    * Higher wide-band IMU noise (sensor + tremor + proprioceptive drift).
    * Occasional micro-stumble spikes on gyro_z (mediolateral instability).
    * Irregular cycle-to-cycle duration (jittered stride time).

Run:
    python generate_synthetic_dataset.py
Output:
    presense_synthetic_gait.csv  (in current working directory)
==============================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass(frozen=True)
class GaitConfig:
    n_cycles: int = 500                # total gait cycles
    fs_hz: int = 100                   # ESP32 sampling rate (MPU6050 window rate)
    mean_cycle_s: float = 1.10         # ~55 strides / min average
    cycle_jitter_s: float = 0.06       # normal stride-time variability
    neuropathy_ratio: float = 0.42     # fraction of cycles labeled 1
    seed: int = 42
    output_csv: str = "presense_synthetic_gait.csv"


# ------------------------------------------------------------------
# Per-cycle waveform synthesis
# ------------------------------------------------------------------
def _phase(n_samples: int) -> np.ndarray:
    """Normalized gait phase in [0, 1) for a single cycle."""
    return np.linspace(0.0, 1.0, n_samples, endpoint=False)


def _gaussian_bump(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    """A Gaussian-shaped bump used to model contact / push-off events."""
    return np.exp(-0.5 * ((phase - center) / width) ** 2)


def _normal_imu(phase: np.ndarray, rng: np.random.Generator) -> dict:
    """Baseline healthy MPU6050 signals for one gait cycle."""
    # Accelerometer (g). Heel strike ~ phase 0.02, toe-off ~ phase 0.62.
    acc_z = (
        1.00                                              # gravity baseline
        + 1.30 * _gaussian_bump(phase, 0.02, 0.035)       # heel strike impact
        + 0.55 * _gaussian_bump(phase, 0.60, 0.055)       # push-off
        - 0.20 * _gaussian_bump(phase, 0.78, 0.10)        # mid-swing dip
    )
    acc_x = (                                             # anterior-posterior
        0.90 * np.sin(2 * np.pi * phase)
        + 0.35 * _gaussian_bump(phase, 0.62, 0.05)        # toe-off thrust
    )
    acc_y = 0.25 * np.sin(2 * np.pi * phase + 0.4)        # mediolateral sway

    # Gyroscope (deg/s). Shank angular velocity peaks during swing.
    gyro_y = (
        260.0 * _gaussian_bump(phase, 0.72, 0.08)         # swing peak
        - 90.0 * _gaussian_bump(phase, 0.05, 0.04)        # heel strike reversal
    )
    gyro_x = 45.0 * np.sin(2 * np.pi * phase + 0.8)
    gyro_z = 30.0 * np.sin(2 * np.pi * phase + 1.6)

    # Sensor noise (MPU6050 typical @ ±2g / ±250 dps configs).
    acc_x += rng.normal(0.0, 0.02, phase.size)
    acc_y += rng.normal(0.0, 0.02, phase.size)
    acc_z += rng.normal(0.0, 0.025, phase.size)
    gyro_x += rng.normal(0.0, 1.5, phase.size)
    gyro_y += rng.normal(0.0, 2.0, phase.size)
    gyro_z += rng.normal(0.0, 1.5, phase.size)

    return {"acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
            "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z}


def _normal_fsr(phase: np.ndarray, rng: np.random.Generator) -> dict:
    """Baseline healthy plantar pressure profile (0-1023 ADC counts)."""
    # Heel loads first, then midfoot, then toes push off.
    heel = 900.0 * _gaussian_bump(phase, 0.08, 0.055)
    mid  = 700.0 * _gaussian_bump(phase, 0.28, 0.075)
    toeL = 820.0 * _gaussian_bump(phase, 0.55, 0.060)
    toeR = 780.0 * _gaussian_bump(phase, 0.57, 0.060)

    for arr in (heel, mid, toeL, toeR):
        arr += rng.normal(0.0, 8.0, phase.size)
        np.clip(arr, 0.0, 1023.0, out=arr)

    return {"fsr_heel": heel, "fsr_toe_left": toeL,
            "fsr_toe_right": toeR, "fsr_mid": mid}


def _normal_altitude(phase: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """BMP180 altitude delta (meters) relative to rolling baseline."""
    # Foot rises during swing, returns to ground on heel strike.
    delta = 0.11 * _gaussian_bump(phase, 0.75, 0.12)
    delta += rng.normal(0.0, 0.004, phase.size)   # baro noise ~ few mm
    return delta


# ------------------------------------------------------------------
# Neuropathy corruption
# ------------------------------------------------------------------
def _apply_neuropathy(cycle: dict, phase: np.ndarray,
                      rng: np.random.Generator) -> None:
    """
    Mutate a normal cycle in-place to reflect hesitation / foot drop:
      * Delay + attenuate FSR ramps (blunted plantar sensation & loading).
      * Weaken dorsiflexion (smaller toe-off IMU peaks).
      * Add wide-band tremor + drift on IMU channels.
      * Inject occasional mediolateral micro-stumble on gyro_z.
    """
    n = phase.size

    # --- FSR: delayed onset and reduced amplitude ---------------------------
    delay = int(rng.uniform(0.08, 0.18) * n)          # shift right ~8-18% cycle
    for key, atten in (("fsr_heel",      0.55),
                       ("fsr_mid",       0.60),
                       ("fsr_toe_left",  0.45),
                       ("fsr_toe_right", 0.45)):
        shifted = np.zeros_like(cycle[key])
        if delay < n:
            shifted[delay:] = cycle[key][:n - delay] * atten
        # Slow, hesitant ramp (low-pass via cumulative smoothing).
        kernel = np.ones(7) / 7.0
        shifted = np.convolve(shifted, kernel, mode="same")
        shifted += rng.normal(0.0, 14.0, n)           # extra sensor noise
        np.clip(shifted, 0.0, 1023.0, out=shifted)
        cycle[key] = shifted

    # --- IMU: reduced dorsiflexion + tremor + drift -------------------------
    # Attenuate toe-off signatures where dorsiflexion is impaired.
    cycle["acc_x"] *= 0.70
    cycle["gyro_y"] *= 0.55                            # weakened swing peak

    # High-frequency tremor (~6-8 Hz) superimposed on gyro channels.
    t = phase  # already normalized; fs baked into cycle length
    tremor_hz = rng.uniform(6.0, 8.5)
    tremor = 12.0 * np.sin(2 * np.pi * tremor_hz * t * (n / 100.0))
    cycle["gyro_x"] += tremor + rng.normal(0.0, 3.5, n)
    cycle["gyro_y"] += rng.normal(0.0, 4.5, n)
    cycle["gyro_z"] += rng.normal(0.0, 3.5, n)

    # Proprioceptive drift on accelerometers.
    drift = np.linspace(0.0, rng.uniform(-0.08, 0.08), n)
    cycle["acc_x"] += drift + rng.normal(0.0, 0.045, n)
    cycle["acc_y"] += rng.normal(0.0, 0.045, n)
    cycle["acc_z"] += rng.normal(0.0, 0.055, n)

    # --- Mediolateral micro-stumble spikes on gyro_z ------------------------
    n_spikes = rng.integers(1, 4)
    for _ in range(n_spikes):
        idx = int(rng.uniform(0.55, 0.95) * n)
        cycle["gyro_z"][idx:min(idx + 3, n)] += rng.choice([-1, 1]) * rng.uniform(60.0, 120.0)


# ------------------------------------------------------------------
# Dataset assembly
# ------------------------------------------------------------------
def generate_dataset(cfg: GaitConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.fs_hz

    frames: list[pd.DataFrame] = []
    t_cursor = 0.0
    altitude_baseline = 0.0                            # slow BMP180 drift

    # Pre-decide labels so the class balance is exact-ish, then shuffle order.
    n_pos = int(round(cfg.n_cycles * cfg.neuropathy_ratio))
    labels = np.array([1] * n_pos + [0] * (cfg.n_cycles - n_pos))
    rng.shuffle(labels)

    for cycle_idx, label in enumerate(labels):
        # Cycle duration (jittered; neuropathic cycles more irregular).
        jitter = cfg.cycle_jitter_s * (2.2 if label == 1 else 1.0)
        duration = max(0.55, rng.normal(cfg.mean_cycle_s, jitter))
        n_samples = max(8, int(round(duration * cfg.fs_hz)))
        phase = _phase(n_samples)

        cycle = {}
        cycle.update(_normal_imu(phase, rng))
        cycle.update(_normal_fsr(phase, rng))
        alt = _normal_altitude(phase, rng)

        if label == 1:
            _apply_neuropathy(cycle, phase, rng)
            # Occasional shuffling gait: reduced foot clearance -> lower altitude peak.
            alt *= rng.uniform(0.35, 0.65)
            alt += rng.normal(0.0, 0.006, n_samples)

        # Absolute timestamps for this cycle
        t = t_cursor + np.arange(n_samples) * dt
        t_cursor += n_samples * dt

        # Slow BMP180 baseline wander (thermal / weather drift).
        altitude_baseline += rng.normal(0.0, 0.0008)
        altitude_delta = alt + altitude_baseline

        df = pd.DataFrame({
            "timestamp":      np.round(t, 4),
            "acc_x":          cycle["acc_x"],
            "acc_y":          cycle["acc_y"],
            "acc_z":          cycle["acc_z"],
            "gyro_x":         cycle["gyro_x"],
            "gyro_y":         cycle["gyro_y"],
            "gyro_z":         cycle["gyro_z"],
            "fsr_heel":       cycle["fsr_heel"].astype(int),
            "fsr_toe_left":   cycle["fsr_toe_left"].astype(int),
            "fsr_toe_right":  cycle["fsr_toe_right"].astype(int),
            "fsr_mid":        cycle["fsr_mid"].astype(int),
            "altitude_delta": np.round(altitude_delta, 5),
            "cycle_id":       cycle_idx,
            "label":          int(label),
        })
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    # Column order matches the ESP32 CSV logger (SD card writer).
    ordered_cols = [
        "timestamp",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "fsr_heel", "fsr_toe_left", "fsr_toe_right", "fsr_mid",
        "altitude_delta",
        "cycle_id",
        "label",
    ]
    return dataset[ordered_cols]


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PreSense synthetic gait dataset.")
    p.add_argument("--cycles", type=int, default=GaitConfig.n_cycles,
                   help="Number of gait cycles to synthesize (default: 500).")
    p.add_argument("--fs", type=int, default=GaitConfig.fs_hz,
                   help="Sampling rate in Hz (default: 100).")
    p.add_argument("--neuropathy-ratio", type=float,
                   default=GaitConfig.neuropathy_ratio,
                   help="Fraction of cycles labeled as foot drop / hesitation (default: 0.42).")
    p.add_argument("--seed", type=int, default=GaitConfig.seed,
                   help="Random seed (default: 42).")
    p.add_argument("--out", type=str, default=GaitConfig.output_csv,
                   help="Output CSV filename.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GaitConfig(
        n_cycles=args.cycles,
        fs_hz=args.fs,
        neuropathy_ratio=args.neuropathy_ratio,
        seed=args.seed,
        output_csv=args.out,
    )

    print(f"[PreSense] Generating {cfg.n_cycles} gait cycles "
          f"@ {cfg.fs_hz} Hz (seed={cfg.seed}) ...")
    df = generate_dataset(cfg)

    out_path = Path(cfg.output_csv).resolve()
    df.to_csv(out_path, index=False)

    # Summary
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    duration_s = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"[PreSense] Wrote {len(df):,} samples across "
          f"{df['cycle_id'].nunique()} cycles -> {out_path}")
    print(f"[PreSense] Class balance : label=0 -> {n_neg:,} samples | "
          f"label=1 -> {n_pos:,} samples")
    print(f"[PreSense] Total recording length: {duration_s:,.2f} s "
          f"(~{duration_s/60:.1f} min)")


if __name__ == "__main__":
    main()
