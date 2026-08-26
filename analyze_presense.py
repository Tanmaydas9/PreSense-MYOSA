"""
analyze_presense.py
==============================================================================
PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
Lead Developer: Tanmay Das
IEEE Sensors Council MYOSA Ideation and Prototype Competition
------------------------------------------------------------------------------
Post-session analysis tool.

Loads the CSV log pulled from the ESP32 SD card (as written by
presense_main.ino's logFrame() / fireNMES() routines) and produces a
multi-panel figure:

    Panel 1 : MPU6050 accelerometer  (acc_x, acc_y, acc_z)      [g]
    Panel 2 : MPU6050 gyroscope      (gyro_x, gyro_y, gyro_z)   [deg/s]
    Panel 3 : FSR plantar pressures  (heel, toe_L, toe_R, mid)  [ADC counts]
    Panel 4 : BMP180 altitude delta                             [m]
    Panel 5 : Inference label timeline (0 = Normal, 1 = Instability)

Every NMES intervention (GPIO 14 HIGH event, logged as an NMES_TRIGGER row)
is drawn as a vertical RED dashed line across ALL panels, with an event
number annotation.

CSV format expected (from presense_main.ino):

    timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,fsr_heel,fsr_toe_left,
    fsr_toe_right,fsr_mid,altitude_delta,label

Event rows interleaved by fireNMES():
    <ms>,EVENT,NMES_TRIGGER,<event_id>

Usage:
    python analyze_presense.py presense_log.csv
    python analyze_presense.py presense_log.csv --save session_plot.png
    python analyze_presense.py presense_log.csv --t0 30000 --t1 45000  # ms zoom
    python analyze_presense.py presense_log.csv --stats-only
==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates  # noqa: F401  (ensures matplotlib backend is healthy)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PlotConfig:
    fsr_channels: tuple = (
        ("fsr_heel",      "FSR Heel (D32)",      "#d62728"),
        ("fsr_toe_left",  "FSR Toe Left (D33)",  "#1f77b4"),
        ("fsr_toe_right", "FSR Toe Right (D34)", "#2ca02c"),
        ("fsr_mid",       "FSR Mid (D35)",       "#9467bd"),
    )
    acc_channels: tuple = (
        ("acc_x", "acc_x [g]"),
        ("acc_y", "acc_y [g]"),
        ("acc_z", "acc_z [g]"),
    )
    gyro_channels: tuple = (
        ("gyro_x", "gyro_x [deg/s]"),
        ("gyro_y", "gyro_y [deg/s]"),
        ("gyro_z", "gyro_z [deg/s]"),
    )
    nmes_color: str = "red"
    nmes_linewidth: float = 1.4
    figure_dpi: int = 130


# ------------------------------------------------------------------
# Log loading
# ------------------------------------------------------------------
def load_log(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the SD-card log into:
        frames : numeric sensor rows (one per 10 ms sample)
        events : NMES_TRIGGER marker rows with absolute timestamps (ms)

    The ESP32 logger emits two row shapes into one file, so we parse
    line-by-line tolerant of either shape.
    """
    frame_rows: list[list] = []
    event_rows: list[tuple] = []

    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()          # discard header line

        for lineno, raw in enumerate(fh, start=2):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")

            # Event marker: "<ms>,EVENT,NMES_TRIGGER,<id>"
            if len(parts) >= 4 and parts[1] == "EVENT":
                try:
                    event_rows.append(
                        (int(parts[0]), parts[2], int(parts[3]))
                    )
                except ValueError:
                    print(f"[WARN] Malformed event row at line {lineno}: {line}")
                continue

            # Frame row: 14 numeric columns
            if len(parts) >= 14:
                try:
                    frame_rows.append([
                        int(parts[0]),                      # timestamp (ms)
                        float(parts[1]),  float(parts[2]),  float(parts[3]),
                        float(parts[4]),  float(parts[5]),  float(parts[6]),
                        int(parts[7]),    int(parts[8]),
                        int(parts[9]),    int(parts[10]),
                        float(parts[11]), int(parts[12]),
                    ])
                except ValueError:
                    print(f"[WARN] Malformed frame row at line {lineno}: {line}")
            else:
                print(f"[WARN] Skipping short row at line {lineno}: {line}")

    frames = pd.DataFrame(frame_rows, columns=[
        "timestamp",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "fsr_heel", "fsr_toe_left", "fsr_toe_right", "fsr_mid",
        "altitude_delta", "label",
    ])

    events = pd.DataFrame(event_rows,
                          columns=["timestamp_ms", "event_type", "event_id"])

    if frames.empty:
        raise ValueError(f"No sensor frames parsed from {csv_path}")

    # Convert to seconds relative to session start for readable axes.
    t0 = frames["timestamp"].iloc[0]
    frames["t_s"] = (frames["timestamp"] - t0) / 1000.0
    if not events.empty:
        events["t_s"] = (events["timestamp_ms"] - t0) / 1000.0

    return frames, events


# ------------------------------------------------------------------
# Session statistics
# ------------------------------------------------------------------
def print_stats(frames: pd.DataFrame, events: pd.DataFrame) -> None:
    duration_s = frames["t_s"].iloc[-1]
    n_samples = len(frames)
    mean_fs = n_samples / duration_s if duration_s > 0 else 0.0

    n_events = len(events)
    n_unstable = int((frames["label"] == 1).sum())
    unstable_pct = 100.0 * n_unstable / max(n_samples, 1)

    print("\n=================== PreSense Session Summary ===================")
    print(f"  Duration              : {duration_s:,.1f} s  ({duration_s/60:.1f} min)")
    print(f"  Samples               : {n_samples:,}")
    print(f"  Effective sample rate : {mean_fs:.1f} Hz")
    print(f"  Instability windows   : {n_unstable:,}  ({unstable_pct:.1f}%)")
    print(f"  NMES interventions    : {n_events}")

    if n_events >= 2:
        intervals = np.diff(events["timestamp_ms"].values) / 1000.0
        print(f"  Inter-event interval  : mean {intervals.mean():.2f} s | "
              f"min {intervals.min():.2f} s | max {intervals.max():.2f} s")

    print("\n  FSR peak pressures (ADC counts):")
    for ch in ("fsr_heel", "fsr_toe_left", "fsr_toe_right", "fsr_mid"):
        print(f"    {ch:<14s}: mean {frames[ch].mean():7.1f} | "
              f"peak {frames[ch].max():4d}")

    print("\n  IMU ranges:")
    for ch in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
        print(f"    {ch:<8s}: [{frames[ch].min():8.2f}, "
              f"{frames[ch].max():8.2f}]")
    print("================================================================\n")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_session(frames: pd.DataFrame,
                 events: pd.DataFrame,
                 cfg: PlotConfig,
                 title_suffix: str = "") -> plt.Figure:
    fig, axes = plt.subplots(
        5, 1,
        figsize=(16, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 2, 1.2]},
    )

    t = frames["t_s"].values

    # ---------------- Panel 1: Accelerometer ----------------------
    ax = axes[0]
    for col, label in cfg.acc_channels:
        ax.plot(t, frames[col], label=label, linewidth=0.9)
    ax.set_ylabel("Acceleration [g]")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.set_title(f"PreSense Session Analysis {title_suffix}".strip())

    # ---------------- Panel 2: Gyroscope --------------------------
    ax = axes[1]
    for col, label in cfg.gyro_channels:
        ax.plot(t, frames[col], label=label, linewidth=0.9)
    ax.set_ylabel("Angular vel [deg/s]")
    ax.legend(loc="upper right", ncol=3, fontsize=9)

    # ---------------- Panel 3: FSR pressure -----------------------
    ax = axes[2]
    for col, label, color in cfg.fsr_channels:
        ax.plot(t, frames[col], label=label, color=color, linewidth=1.1)
    ax.set_ylabel("FSR [ADC counts]")
    ax.set_ylim(-30, 1080)
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    # ---------------- Panel 4: Altitude delta ---------------------
    ax = axes[3]
    ax.plot(t, frames["altitude_delta"], color="#17becf",
            linewidth=1.0, label="altitude_delta [m]")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Alt delta [m]")
    ax.legend(loc="upper right", fontsize=9)

    # ---------------- Panel 5: Inference label --------------------
    ax = axes[4]
    ax.fill_between(t, frames["label"].values, step="post",
                    color="#ff7f0e", alpha=0.55, label="label (1=instability)")
    ax.set_ylabel("ML label")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Instab."])
    ax.set_xlabel("Session time [s]")
    ax.legend(loc="upper right", fontsize=9)

    # ------------- NMES intervention overlays (all panels) --------
    if not events.empty:
        for _, ev in events.iterrows():
            for ax in axes:
                ax.axvline(ev["t_s"],
                           color=cfg.nmes_color,
                           linewidth=cfg.nmes_linewidth,
                           linestyle="--",
                           alpha=0.85, zorder=10)
            # Annotate event number on the top panel
            axes[0].annotate(
                f"NMES #{int(ev['event_id'])}",
                xy=(ev["t_s"], axes[0].get_ylim()[1]),
                xytext=(4, -2), textcoords="offset points",
                rotation=90, va="top", fontsize=8,
                color=cfg.nmes_color, fontweight="bold",
            )
    else:
        axes[0].text(0.01, 0.92, "No NMES events in log",
                     transform=axes[0].transAxes, color="gray",
                     fontsize=10, style="italic")

    # Shared legend entry for the NMES lines
    axes[0].plot([], [], color=cfg.nmes_color, linestyle="--",
                 linewidth=cfg.nmes_linewidth,
                 label="NMES trigger (D14)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", ncol=2, fontsize=9)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-session analysis of a PreSense ESP32 SD-card log.")
    p.add_argument("csv", type=str,
                   help="Path to presense_log.csv pulled from the SD card.")
    p.add_argument("--save", type=str, default=None,
                   help="Save figure to this path instead of showing it "
                        "(e.g. session_plot.png).")
    p.add_argument("--t0", type=float, default=None,
                   help="Zoom window start, in seconds of session time.")
    p.add_argument("--t1", type=float, default=None,
                   help="Zoom window end, in seconds of session time.")
    p.add_argument("--stats-only", action="store_true",
                   help="Print session statistics without plotting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"[ERROR] Log file not found: {csv_path}")

    print(f"[PreSense] Loading {csv_path} ...")
    frames, events = load_log(csv_path)
    print(f"[PreSense] Parsed {len(frames):,} sensor frames and "
          f"{len(events)} NMES events.")

    # Optional zoom window
    if args.t0 is not None or args.t1 is not None:
        lo = args.t0 if args.t0 is not None else frames["t_s"].iloc[0]
        hi = args.t1 if args.t1 is not None else frames["t_s"].iloc[-1]
        mask = (frames["t_s"] >= lo) & (frames["t_s"] <= hi)
        frames = frames.loc[mask].reset_index(drop=True)
        if not events.empty:
            events = events.loc[
                (events["t_s"] >= lo) & (events["t_s"] <= hi)
            ].reset_index(drop=True)
        print(f"[PreSense] Zoomed to t = [{lo:.1f}, {hi:.1f}] s : "
              f"{len(frames):,} frames, {len(events)} events.")

    print_stats(frames, events)

    if args.stats_only:
        return

    cfg = PlotConfig()
    suffix = (f"(zoom {args.t0:.0f}-{args.t1:.0f} s)"
              if (args.t0 is not None and args.t1 is not None) else "")
    fig = plot_session(frames, events, cfg, title_suffix=suffix)

    if args.save:
        out = Path(args.save).resolve()
        fig.savefig(out, dpi=cfg.figure_dpi, bbox_inches="tight")
        print(f"[PreSense] Figure saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
