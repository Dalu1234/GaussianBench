# Deprecated scenes

Scenes in this directory were superseded in later benchmark versions. They
are preserved, with their frozen pass_criteria, provenance blocks, and
initial-position files, so that results scored against them remain
reproducible historical records. They are NOT part of the current
benchmark suite: run_all.py only discovers scenes in scenes/ itself, never
in this directory, and new submissions should not target them.

| scene | deprecated in | superseded by | reason |
|---|---|---|---|
| a1_conservation_v1 | v0.2.0 | a1a_conservation_inplace_v1 + a1b_conservation_drifting_v1 | Conflated two distinct diagnostics (transfer-scheme conservation and domain handling of a far-translating body); split so each is tested in isolation. |

The positions file a1_ball_positions.npy is duplicated here so the
deprecated scene stays self-contained; the copy in scenes/ is the live one
used by a1a and a1b (all three scenes share identical particles).
