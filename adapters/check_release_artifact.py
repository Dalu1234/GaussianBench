"""Audit the packaged external-entry evidence without rerunning simulators.

This checks provenance completeness and anonymity only. It deliberately does
not recompute or alter any score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROFILES = {
    "PhysGaussian_native_2026-08-27": "8339ed6a",
    "GaussianFluent_full": "de7e1067",
    "OmniPhysGS_full": "5ab915b0",
    "PhysDreamer_full": "95beb71",
    "Physics3D_broad": "32d9218",
    "GASP_broad": "186c9c9",
}
REQUIRED_META = {
    "system_name",
    "system_version",
    "spec_version",
    "columns_present",
    "wall_clock_seconds",
    "hardware",
}
ALLOWED_STATUS = {"PASS", "FAIL", "INVALID", "NA", "ERROR"}
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\|/Users/|/home/)", re.I)


def default_reports_root() -> Path:
    script = Path(__file__).resolve()
    repository = script.parents[1]
    candidates = (
        repository / "reports",
        script.parents[3] / "reports",
        repository / "results",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def audit(reports_root: Path) -> list[str]:
    errors: list[str] = []
    for profile, pin_prefix in PROFILES.items():
        directory = reports_root / profile
        report_file = directory / "report.json"
        if not report_file.is_file():
            errors.append(f"{profile}: missing report.json")
            continue
        try:
            rows = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as exc:  # reviewer-facing diagnostic
            errors.append(f"{profile}: invalid report.json ({exc})")
            continue
        if len(rows) != 18:
            errors.append(f"{profile}: expected 18 scene rows, found {len(rows)}")
        for row in rows:
            status = row.get("status")
            if status not in ALLOWED_STATUS:
                errors.append(
                    f"{profile}/{row.get('scene_id', '?')}: invalid status {status!r}"
                )

        meta_files = sorted(directory.glob("*/meta.json"))
        if not meta_files:
            errors.append(f"{profile}: no per-scene meta.json files")
        versions: set[str] = set()
        for meta_file in meta_files:
            text = meta_file.read_text(encoding="utf-8")
            if LOCAL_PATH.search(text):
                errors.append(f"{profile}/{meta_file.parent.name}: local path leaked")
            try:
                meta = json.loads(text)
            except Exception as exc:
                errors.append(f"{profile}/{meta_file.parent.name}: invalid JSON ({exc})")
                continue
            missing = sorted(REQUIRED_META - set(meta))
            if missing:
                errors.append(
                    f"{profile}/{meta_file.parent.name}: missing {', '.join(missing)}"
                )
            version = str(meta.get("system_version", ""))
            if version:
                versions.add(version)
                if not version.startswith(pin_prefix):
                    errors.append(
                        f"{profile}/{meta_file.parent.name}: version {version!r} "
                        f"does not match pin {pin_prefix}"
                    )
        print(
            f"{profile}: {len(rows)} report rows, {len(meta_files)} metadata files, "
            f"versions={sorted(versions)}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=default_reports_root())
    args = parser.parse_args()
    errors = audit(args.reports_root.resolve())
    if errors:
        print("\nAUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nAUDIT PASSED: provenance fields, pins, statuses, and local-path scan are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
