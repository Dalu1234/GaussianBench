"""Run every supported GaussianBench scene through the exporter, one
subprocess per scene.

    .\\.venv\\Scripts\\python.exe -m export.run_all_scenes
        [--out bench_results] [--scenes-dir <gaussianbench>/scenes]
        [--only a1a,a2,...] [--max-seconds S]

One subprocess per scene is mandatory, not a convenience: the simulator
fixes N_PARTICLES and the Taichi field shapes at import time from CLI
flags, so scenes with different particle counts cannot share a process.
"""

import argparse
import os
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
DEFAULT_SCENES_DIR = Path(
    os.environ.get("GAUSSIANBENCH_SCENES", REPO.parent / "gaussianbench" / "scenes")
)

# scene file stem -> extra exporter flags
LOGGER = logging.getLogger(__name__)

# scene file stem -> extra exporter flags
SCENES = {
    "a1a_conservation_inplace_v1": [],
    "a1b_conservation_drifting_v1": [],
    "a2_beam_frequency_v1": [],
    "a3_wavespeed_v1": [],
    "a4_meltfront_v1": [],
    "a5_stiffness_v1": [],
    "c1_covariance_v1": ["--tier-c"],
    "c2_blobhealth_melt_v1": ["--tier-c"],
    "c2_blobhealth_shear_v1": ["--tier-c"],
    "c3_roundtrip_v1": ["--tier-c"],
    "c4_captured_frequency_v1": [],
}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "bench_results")
    ap.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIR)
    ap.add_argument("--only", default=None,
                    help="Comma-separated scene-id prefixes to run.")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    wanted = args.only.split(",") if args.only else None
    results = {}
    for stem, flags in SCENES.items():
        if wanted and not any(stem.startswith(w) for w in wanted):
            continue
        scene_path = args.scenes_dir / f"{stem}.json"
        if not scene_path.exists():
            results[stem] = "scene file missing"
            continue
        cmd = [sys.executable, "-m", "export.run_bench_scene",
               "--scene", str(scene_path), "--out", str(args.out)] + flags
        if args.max_seconds is not None:
            cmd += ["--max-seconds", str(args.max_seconds)]
        LOGGER.info("\\n===== %s =====", stem)
        proc = subprocess.run(cmd, cwd=str(REPO))
        results[stem] = "ok" if proc.returncode == 0 else \
            f"exit {proc.returncode}"

    LOGGER.info("\\n===== batch summary =====")
    for stem, status in results.items():
        LOGGER.info("  %-20s %s", status, stem)
    return 1 if any(v != "ok" for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
