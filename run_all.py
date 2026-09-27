"""Run every applicable GaussianBench scorer against a results directory.

Usage:
    python run_all.py --results-dir results/example --scenes-dir scenes/

For each scene JSON in scenes-dir, looks for results-dir/<scene_id>/ and runs
the scorer named in the scene's "scorer" key. Emits report.json and report.md
into results-dir. Missing submissions are reported NA (with a note), never
silently skipped; a malformed submission is ERROR; a submission that violates
the scene's regime assumptions is INVALID.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scoring import common


def discover_scenes(scenes_dir: Path) -> list[dict]:
    scenes = []
    # Non-recursive on purpose: scenes/deprecated/ holds superseded scenes
    # kept only for historical reproducibility and must not be discovered.
    for path in sorted(scenes_dir.glob("*.json")):
        try:
            scenes.append(common.load_scene(path))
        except common.SubmissionError as e:
            print(f"[skip] {path.name}: {e}", file=sys.stderr)
    return scenes


def run_one(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    scene_results = results_dir / scene["scene_id"]
    if not scene_results.exists():
        note = ("No submission directory found. For capability tests this "
                "means the system does not support the capability.")
        return common.verdict_from_scene(scene, common.STATUS_NA, notes=[note])

    scorer_name = scene.get("scorer")
    if not scorer_name:
        return common.verdict_from_scene(
            scene, common.STATUS_ERROR,
            notes=[f"Scene {scene['scene_id']} declares no 'scorer' key."])
    try:
        mod = importlib.import_module(f"scoring.{scorer_name}")
    except ImportError as e:
        return common.verdict_from_scene(
            scene, common.STATUS_ERROR,
            notes=[f"Scorer module scoring.{scorer_name} not importable: {e}"])

    try:
        return mod.score(scene, scene_results, scenes_dir)
    except common.SubmissionError as e:
        return common.verdict_from_scene(
            scene, common.STATUS_ERROR, notes=[str(e)])
    except Exception:
        return common.verdict_from_scene(
            scene, common.STATUS_ERROR,
            notes=["Scorer crashed. This is a benchmark bug, please report it.",
                   traceback.format_exc()])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--scenes-dir", default=Path(__file__).parent / "scenes",
                    type=Path)
    args = ap.parse_args(argv)

    scenes = discover_scenes(args.scenes_dir)
    if not scenes:
        print(f"No scene JSONs found in {args.scenes_dir}", file=sys.stderr)
        return 2

    verdicts = []
    for scene in scenes:
        v = run_one(scene, args.results_dir, args.scenes_dir)
        verdicts.append(v)
        print(f"{v.status:8s} {v.scene_id}")

    common.generate_report(verdicts, args.results_dir)
    print(f"\nReport written to {args.results_dir / 'report.md'} "
          f"and {args.results_dir / 'report.json'}")
    # Exit code communicates only infrastructure health, not pass/fail:
    # a FAIL is a valid benchmark outcome, an ERROR means something is broken.
    return 1 if any(v.status == common.STATUS_ERROR for v in verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
