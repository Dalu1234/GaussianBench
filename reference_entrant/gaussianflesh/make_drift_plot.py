"""
make_drift_plot.py
Generate the §4.4 figure for the GaussianFlesh CVPR camera-ready.

Reads a per-frame drift CSV (drift_*.csv, schema from evaluate_drift.py) and
emits a two-panel PDF: drift_mean over frames (top), J_min over frames
(bottom), both panels showing TL (ours) and UL (baseline) lines.

Usage:
    python make_drift_plot.py [csv_path] [output_pdf_path]

Defaults to the 3000-frame run from 2026-05-17 and writes into results/.
"""
import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
CSV_DEFAULT = REPO / "results" / "drift_20260517_010003.csv"
OUT_DEFAULT = REPO / "results" / "tl_vs_ul_drift.pdf"

# Colors matching the compare-mode viewport: red for TL (ours), blue for UL
TL_COLOR = "#e63946"
UL_COLOR = "#457b9d"
LOGGER = logging.getLogger(__name__)


def load_csv(path: Path):
    """Returns {solver_name: {column: np.array}}."""
    rows_by_solver = defaultdict(list)
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows_by_solver[r["solver_type"]].append(r)
    out = {}
    for solver, rows in rows_by_solver.items():
        rows.sort(key=lambda r: int(r["frame"]))
        out[solver] = {
            "frame":          np.array([int(r["frame"])         for r in rows]),
            "drift_mean":     np.array([float(r["drift_mean"])  for r in rows]),
            "drift_max":      np.array([float(r["drift_max"])   for r in rows]),
            "J_min":          np.array([float(r["J_min"])       for r in rows]),
            "J_max":          np.array([float(r["J_max"])       for r in rows]),
            "Uyy_spread":     np.array([float(r["Uyy_spread"])  for r in rows]),
            "sigma_min_mean": np.array([float(r["sigma_min_mean"]) for r in rows]),
        }
    return out


def plot(data: dict, out: Path) -> None:
    """Two-panel stacked figure for the CVPR §4.4 results."""
    # CVPR is two-column 10pt; \linewidth ≈ 3.3 inches.  Render at higher
    # nominal size and let LaTeX scale  -  keeps text legible after scaling.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.4, 3.9), sharex=True,
        gridspec_kw=dict(hspace=0.32),
    )

    tl = data["tl_apic"]
    ul = data["ul_mlsmpm"]

    # ---- top panel: drift_mean -----------------------------------------
    ax_top.plot(tl["frame"], tl["drift_mean"], color=TL_COLOR, linewidth=1.2,
                label="TL-APIC (ours)")
    ax_top.plot(ul["frame"], ul["drift_mean"], color=UL_COLOR, linewidth=1.2,
                label="UL-MLS-MPM (baseline)")
    # Rest-state baseline reference
    ax_top.axhline(y=0.07885, color="gray", linewidth=0.7,
                   linestyle="--", alpha=0.6, zorder=0)
    ax_top.set_title("Mesh-to-isosurface drift (mean)", fontsize=9, pad=4)
    ax_top.set_ylabel("drift mean (m)", fontsize=8)
    ax_top.legend(fontsize=7, loc="upper left", frameon=False,
                  handlelength=1.3, borderpad=0.2)
    ax_top.tick_params(labelsize=7)
    ax_top.grid(True, alpha=0.25, linewidth=0.4)
    ax_top.set_ylim(0.072, 0.100)
    # Annotate the final-frame gap
    last_frame = tl["frame"][-1]
    ax_top.annotate(
        f"{(ul['drift_mean'][-1] / tl['drift_mean'][-1] - 1) * 100:+.0f}%",
        xy=(last_frame, ul["drift_mean"][-1]),
        xytext=(-25, -10), textcoords="offset points",
        fontsize=7, color=UL_COLOR,
    )

    # ---- bottom panel: J_min -------------------------------------------
    ax_bot.plot(tl["frame"], tl["J_min"], color=TL_COLOR, linewidth=1.2)
    ax_bot.plot(ul["frame"], ul["J_min"], color=UL_COLOR, linewidth=1.2)
    ax_bot.axhline(y=1.0, color="gray", linewidth=0.7,
                   linestyle="--", alpha=0.6, zorder=0)
    ax_bot.set_title("Per-particle volume ratio (min)", fontsize=9, pad=4)
    ax_bot.set_xlabel("frame", fontsize=8)
    ax_bot.set_ylabel(r"$J_{\min}$", fontsize=9)
    ax_bot.tick_params(labelsize=7)
    ax_bot.grid(True, alpha=0.25, linewidth=0.4)
    ax_bot.set_ylim(0.3, 1.05)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    LOGGER.info("wrote %s  (%.1f KB)", out, out.stat().st_size / 1024)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=CSV_DEFAULT)
    parser.add_argument("output_pdf_path", nargs="?", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)

    csv_path = args.csv_path
    out_path = args.output_pdf_path
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    data = load_csv(csv_path)
    expected = {"tl_apic", "ul_mlsmpm"}
    missing = expected - data.keys()
    if missing:
        sys.exit(f"CSV missing solver(s): {missing}.  Re-run --scene=compare first.")
    plot(data, out_path)


if __name__ == "__main__":
    main()
