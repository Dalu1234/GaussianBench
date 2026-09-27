"""Create a controlled frozen-covariance fault from an existing Tier C entry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source/"trajectory.parquet", output/"trajectory.parquet")
    covariance = pd.read_parquet(source/"covariances.parquet")
    rest = covariance[covariance.frame == covariance.frame.min()].copy()
    frames = sorted(covariance.frame.unique())
    frozen = []
    for frame in frames:
        block = rest.copy(); block["frame"] = frame; frozen.append(block)
    pd.concat(frozen, ignore_index=True).to_parquet(output/"covariances.parquet", index=False)
    meta = json.loads((source/"meta.json").read_text(encoding="utf-8"))
    meta["system_name"] = meta["system_name"] + " frozen-covariance fault"
    notes = meta.get("notes", "")
    meta["notes"] = (str(notes) + " | CONTROLLED FAULT: covariance held at frame-zero values; "
                     "trajectory and entrant-native F are unchanged.")
    (output/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
