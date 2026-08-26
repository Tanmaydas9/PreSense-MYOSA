"""
track_rehabilitation_progress.py
==============================================================================
PreSense - Predictive Sensory Augmentation Platform for Peripheral Neuropathy
Lead Developer: Tanmay Das
IEEE Sensors Council MYOSA Ideation and Prototype Competition
------------------------------------------------------------------------------
Longitudinal rehabilitation tracker.

Scans a directory containing multiple PreSense session logs (CSV files pulled
from the ESP32 SD card, in the format written by presense_main.ino) and, for
each session, computes:

    1. Gait Confidence Score (0-100)
       Proportion of session time classified as NORMAL gait (label == 0),
       expressed as a percentage. Higher is better.

    2. NMES Intervention Frequency (events / minute)
       Count of NMES_TRIGGER event rows divided by session duration.
       Lower is better (system needs to intervene less often).

    3. Toe-Clearance Efficiency (0-100)
       Mean of per-swing-cycle peak altitude_delta (foot clearance during
       swing), normalized against a healthy reference clearance.
       Higher is better (less shuffling / foot drop).

Sessions are ordered chronologically by:
    (a) a date parsed from the filename (e.g. presense_log_2026-08-01.csv,
        session_20260812.csv), falling back to
    (b) file modification time.

Outputs:
    * A Matplotlib clinical summary dashboard (trend plots + summary panel)
    * A per-session metrics table printed to stdout
    * Optional CSV export of the per-session metrics

Usage:
    python track_rehabilitation_progress.py ./session_logs/
    python track_rehabilitation_progress.py ./session_logs/ \
            --save dashboard.png --export metrics.csv
    python track_rehabilitation_progress.py ./session_logs/ --patient "ID-042"
==============================================================================
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ClinicalConfig:
    # Healthy-reference swing-phase foot clearance (meters).
    # Used to normalize toe-clearance efficiency to 0-100.
    reference_clearance_m: float = 0.10
    # Minimum samples for a session to be considered valid.
    min_samples: int = 100
    # Heel-contact threshold for swing/stance segmentation (ADC counts).
    heel_contact_threshold: int = 260
    # Dashboard cosmetics
    figure_dpi: int = 130
    confidence_color: str = "#2ca02c"
    nmes_color: str = "#d62728"
    clearance_color: str = "#1f77b4"


# ------------------------------------------------------------------
# Per-session metric container
# ------------------------------------------------------------------
@dataclass
class SessionMetrics:
    file: Path
    session_dt: datetime
    duration_s: float
    n_samples: int
    gait_confidence: float          # 0-100, higher is better
    nmes_per_minute: float          # lower is better
    n_nmes_events: int
    toe_clearance_efficiency: float # 0-100, higher is better
    mean_peak_clearance_m: float
    n_swing_cycles: int
    warnings: list = field(default_factory=list)


# ------------------------------------------------------------------
# Log parsing (same dual-format parser as analyze_presense.py)
# ------------------------------------------------------------------
def load_log(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frames, events) from one PreSense SD-card log."""
    frame_rows: list[list] = []
    event_rows: list[tuple] = []

    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()  # header
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4 and parts[1] == "EVENT":
                try:
                    event_rows.append((int(parts[0]), parts[2], int(parts[3])))
                except ValueError:
                    continue
            elif len(parts) >= 14:
                try:
                    frame_rows.append([
                        int(parts[0]),
                        float(parts[1]), float(parts[2]), float(parts[3]),
                        float(parts[4]), float(parts[5]), float(parts[6]),
                        int(parts[7]), int(parts[8]), int(parts[9]),
                        int(parts[10]), float(parts[11]), int(parts[12]),
                    ])
                except ValueError:
                    continue

    frames = pd.DataFrame(frame_rows, columns=[
        "timestamp", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "fsr_heel", "fsr_toe_left", "fsr_toe_right", "fsr_mid",
        "altitude_delta", "label",
    ])
    events = pd.DataFrame(event_rows,
                          columns=["timestamp_ms", "event_type", "event_id"])
    return frames, events


# ------------------------------------------------------------------
# Session dating
# ------------------------------------------------------------------
_DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})"),   # 2026-08-01 / 2026_08_01
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),           # 20260801
]

def session_datetime(path: Path) -> datetime:
    """Best-effort session date: filename first, file mtime as fallback."""
    for pat in _DATE_PATTERNS:
        m = pat.search(path.stem)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)),
                                int(m.group(3)))
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime)


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------
def compute_session_metrics(path: Path,
                            cfg: ClinicalConfig) -> SessionMetrics | None:
    frames, events = load_log(path)
    warns: list[str] = []

    if len(frames) < cfg.min_samples:
        print(f"[SKIP] {path.name}: only {len(frames)} samples "
              f"(< {cfg.min_samples})")
        return None

    t = frames["timestamp"].values
    duration_s = max((t[-1] - t[0]) / 1000.0, 1.0)
    n_samples = len(frames)

    # Effective sample-rate sanity check (firmware target: 100 Hz)
    eff_fs = n_samples / duration_s
    if eff_fs < 70.0:
        warns.append(f"low effective sample rate ({eff_fs:.0f} Hz)")

    # ---- Metric 1: Gait Confidence Score -------------------------------
    n_normal = int((frames["label"] == 0).sum())
    gait_confidence = 100.0 * n_normal / n_samples

    # ---- Metric 2: NMES interventions per minute -----------------------
    n_events = len(events)
    nmes_per_min = n_events / (duration_s / 60.0)

    # ---- Metric 3: Toe-Clearance Efficiency ----------------------------
    # Segment swing phases: contiguous runs where heel FSR is unloaded
    # (foot in the air). Peak altitude_delta within each swing = clearance.
    heel = frames["fsr_heel"].values
    alt = frames["altitude_delta"].values
    in_swing = heel < cfg.heel_contact_threshold

    peaks: list[float] = []
    run_start = None
    min_swing_samples = max(int(0.20 * eff_fs), 5)   # ignore <200 ms blips
    for i, sw in enumerate(in_swing):
        if sw and run_start is None:
            run_start = i
        elif not sw and run_start is not None:
            if i - run_start >= min_swing_samples:
                peaks.append(float(np.max(alt[run_start:i])))
            run_start = None
    if run_start is not None and n_samples - run_start >= min_swing_samples:
        peaks.append(float(np.max(alt[run_start:])))

    if peaks:
        mean_peak = float(np.mean(peaks))
        efficiency = 100.0 * min(mean_peak / cfg.reference_clearance_m, 1.0)
        n_cycles = len(peaks)
    else:
        mean_peak = 0.0
        efficiency = 0.0
        n_cycles = 0
        warns.append("no valid swing cycles segmented")

    return SessionMetrics(
        file=path,
        session_dt=session_datetime(path),
        duration_s=duration_s,
        n_samples=n_samples,
        gait_confidence=gait_confidence,
        nmes_per_minute=nmes_per_min,
        n_nmes_events=n_events,
        toe_clearance_efficiency=efficiency,
        mean_peak_clearance_m=mean_peak,
        n_swing_cycles=n_cycles,
        warnings=warns,
    )


# ------------------------------------------------------------------
# Dashboard rendering
# ------------------------------------------------------------------
def _trendline(ax, x_num: np.ndarray, y: np.ndarray, color: str) -> None:
    """Least-squares trendline over session sequence."""
    if len(x_num) < 2:
        return
    coef = np.polyfit(x_num, y, 1)
    ax.plot(x_num, np.polyval(coef, x_num),
            color=color, linestyle=":", linewidth=2.0, alpha=0.9,
            label=f"trend ({coef[0]:+.2f}/session)")


def _delta_text(values: np.ndarray, higher_is_better: bool) -> str:
    """First->last change with a direction arrow and clinical verdict."""
    if len(values) < 2:
        return "n/a"
    delta = values[-1] - values[0]
    improved = (delta > 0) if higher_is_better else (delta < 0)
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    verdict = "improving" if improved else ("declining" if delta != 0 else "stable")
    return f"{arrow} {abs(delta):.1f}  ({verdict})"


def render_dashboard(sessions: list[SessionMetrics],
                     cfg: ClinicalConfig,
                     patient_label: str,
                     save_path: Path | None) -> None:
    x = np.arange(len(sessions))
    dates = [s.session_dt for s in sessions]
    date_labels = [d.strftime("%b %d") for d in dates]

    conf = np.array([s.gait_confidence for s in sessions])
    nmes = np.array([s.nmes_per_minute for s in sessions])
    clear = np.array([s.toe_clearance_efficiency for s in sessions])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9],
                          hspace=0.38, wspace=0.22)

    fig.suptitle(
        f"PreSense — Rehabilitation Progress Dashboard"
        + (f"  |  Patient: {patient_label}" if patient_label else "")
        + f"  |  {len(sessions)} sessions "
          f"({dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')})",
        fontsize=15, fontweight="bold")

    # ---------------- Panel 1: Gait Confidence ---------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, conf, "o-", color=cfg.confidence_color, linewidth=2,
             markersize=7, label="gait confidence")
    _trendline(ax1, x, conf, cfg.confidence_color)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x); ax1.set_xticklabels(date_labels, rotation=45, ha="right")
    ax1.set_ylabel("Score (0–100)")
    ax1.set_title("Gait Confidence Score  (higher = more normal gait)", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.text(0.02, 0.90, _delta_text(conf, higher_is_better=True),
             transform=ax1.transAxes, fontsize=10, fontweight="bold",
             color=cfg.confidence_color)

    # ---------------- Panel 2: NMES frequency ----------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, nmes, color=cfg.nmes_color, alpha=0.75, label="NMES/min")
    _trendline(ax2, x, nmes, cfg.nmes_color)
    ax2.set_xticks(x); ax2.set_xticklabels(date_labels, rotation=45, ha="right")
    ax2.set_ylabel("Interventions / minute")
    ax2.set_title("NMES Intervention Frequency  (lower = less assistance needed)",
                  fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.text(0.02, 0.90, _delta_text(nmes, higher_is_better=False),
             transform=ax2.transAxes, fontsize=10, fontweight="bold",
             color=cfg.nmes_color)

    # ---------------- Panel 3: Toe-clearance efficiency ------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, clear, "s-", color=cfg.clearance_color, linewidth=2,
             markersize=7, label="toe-clearance efficiency")
    _trendline(ax3, x, clear, cfg.clearance_color)
    ax3.set_ylim(0, 105)
    ax3.set_xticks(x); ax3.set_xticklabels(date_labels, rotation=45, ha="right")
    ax3.set_ylabel("Efficiency (0–100)")
    ax3.set_title("Toe-Clearance Efficiency  (higher = better swing clearance)",
                  fontsize=11)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.text(0.02, 0.90, _delta_text(clear, higher_is_better=True),
             transform=ax3.transAxes, fontsize=10, fontweight="bold",
             color=cfg.clearance_color)

    # ---------------- Panel 4: Composite score ---------------------
    ax4 = fig.add_subplot(gs[1, 1])
    # Composite: confidence + clearance rewarded, NMES frequency penalized
    # (normalized to the best session's rate).
    nmes_norm = nmes / max(nmes.max(), 1e-6)
    composite = 0.45 * conf + 0.35 * clear + 0.20 * (100.0 * (1.0 - nmes_norm))
    ax4.plot(x, composite, "D-", color="#9467bd", linewidth=2.2,
             markersize=7, label="composite mobility index")
    _trendline(ax4, x, composite, "#9467bd")
    ax4.set_ylim(0, 105)
    ax4.set_xticks(x); ax4.set_xticklabels(date_labels, rotation=45, ha="right")
    ax4.set_ylabel("Index (0–100)")
    ax4.set_title("Composite Mobility Index  (0.45·conf + 0.35·clear + 0.20·NMES-free)",
                  fontsize=11)
    ax4.legend(fontsize=8, loc="lower right")
    ax4.text(0.02, 0.90, _delta_text(composite, higher_is_better=True),
             transform=ax4.transAxes, fontsize=10, fontweight="bold",
             color="#9467bd")

    # ---------------- Panel 5: Clinical summary table --------------
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    col_labels = ["Session", "Date", "Duration", "Confidence",
                  "NMES/min", "Events", "Clearance Eff.", "Cycles", "Flags"]
    cell_text = []
    for i, s in enumerate(sessions):
        cell_text.append([
            f"#{i+1}",
            s.session_dt.strftime("%Y-%m-%d"),
            f"{s.duration_s/60:.1f} min",
            f"{s.gait_confidence:.1f}",
            f"{s.nmes_per_minute:.2f}",
            f"{s.n_nmes_events}",
            f"{s.toe_clearance_efficiency:.1f}",
            f"{s.n_swing_cycles}",
            "; ".join(s.warnings) if s.warnings else "—",
        ])

    table = ax5.table(cellText=cell_text, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#39424e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best/worst confidence sessions
    if len(sessions) >= 2:
        best_i = int(np.argmax(conf)); worst_i = int(np.argmin(conf))
        for j in range(len(col_labels)):
            table[best_i + 1, j].set_facecolor("#e6f4ea")
            table[worst_i + 1, j].set_facecolor("#fdecea")

    ax5.set_title("Per-Session Clinical Summary  "
                  "(green = best confidence, red = worst)",
                  fontsize=11, pad=14)

    if save_path:
        fig.savefig(save_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        print(f"[PreSense] Dashboard saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track PreSense rehabilitation progress across sessions.")
    p.add_argument("log_dir", type=str,
                   help="Directory containing session CSV logs.")
    p.add_argument("--pattern", type=str, default="*.csv",
                   help="Glob pattern for log files (default: *.csv).")
    p.add_argument("--save", type=str, default=None,
                   help="Save dashboard to this path (e.g. dashboard.png).")
    p.add_argument("--export", type=str, default=None,
                   help="Export per-session metrics to this CSV path.")
    p.add_argument("--patient", type=str, default="",
                   help="Patient identifier shown on the dashboard title.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClinicalConfig()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {log_dir}")

    csv_files = sorted(log_dir.glob(args.pattern))
    if not csv_files:
        sys.exit(f"[ERROR] No files matching '{args.pattern}' in {log_dir}")

    print(f"[PreSense] Found {len(csv_files)} session log(s) in {log_dir}")

    sessions: list[SessionMetrics] = []
    for path in csv_files:
        m = compute_session_metrics(path, cfg)
        if m is not None:
            sessions.append(m)
            flag_str = f"  [WARN: {'; '.join(m.warnings)}]" if m.warnings else ""
            print(f"  {path.name:<38s} conf={m.gait_confidence:5.1f}  "
                  f"nmes/min={m.nmes_per_minute:5.2f}  "
                  f"clear={m.toe_clearance_efficiency:5.1f}{flag_str}")

    if len(sessions) < 1:
        sys.exit("[ERROR] No valid sessions parsed.")

    # Chronological ordering
    sessions.sort(key=lambda s: s.session_dt)

    # ---- Metrics table export --------------------------------------
    df = pd.DataFrame([{
        "session": i + 1,
        "date": s.session_dt.strftime("%Y-%m-%d"),
        "file": s.file.name,
        "duration_s": round(s.duration_s, 1),
        "n_samples": s.n_samples,
        "gait_confidence": round(s.gait_confidence, 2),
        "nmes_per_minute": round(s.nmes_per_minute, 3),
        "n_nmes_events": s.n_nmes_events,
        "toe_clearance_efficiency": round(s.toe_clearance_efficiency, 2),
        "mean_peak_clearance_m": round(s.mean_peak_clearance_m, 4),
        "n_swing_cycles": s.n_swing_cycles,
    } for i, s in enumerate(sessions)])

    print("\n================= Longitudinal Metrics =================")
    print(df.to_string(index=False))
    print("=========================================================")

    if len(sessions) >= 2:
        print(f"  Confidence Δ (first→last) : "
              f"{df['gait_confidence'].iloc[-1] - df['gait_confidence'].iloc[0]:+.1f}")
        print(f"  NMES/min Δ   (first→last) : "
              f"{df['nmes_per_minute'].iloc[-1] - df['nmes_per_minute'].iloc[0]:+.2f}")
        print(f"  Clearance Δ  (first→last) : "
              f"{df['toe_clearance_efficiency'].iloc[-1] - df['toe_clearance_efficiency'].iloc[0]:+.1f}")

    if args.export:
        out = Path(args.export).resolve()
        df.to_csv(out, index=False)
        print(f"[PreSense] Metrics exported -> {out}")

    render_dashboard(
        sessions, cfg,
        patient_label=args.patient,
        save_path=Path(args.save).resolve() if args.save else None,
    )


if __name__ == "__main__":
    main()

# Organize SD-card pulls one per session, e.g.:
#   session_logs/presense_log_2026-08-01.csv
#   session_logs/presense_log_2026-08-08.csv
#   ...

python track_rehabilitation_progress.py ./session_logs \
    --patient "DPN-042" --save dashboard.png --export metrics.csv
