"""
duck_diagnostic.py
Post-submission diagnostic: separate head/body/neck behavior in the rubber-duck
sim to figure out whether the visual head-vs-body distinction is real physics,
mesh-skin artifact, or cohesion failure.

NO simulation code is modified.  This script imports gaussian_mesh_poc, tags
particles by their rest-position ellipsoid membership, runs the existing duck
drop, and logs per-region diagnostics to CSV.

Run as:
    python duck_diagnostic.py            # Step 2 only  -  show tag counts and stop
    python duck_diagnostic.py --go       # Steps 2 + 3 + 4  -  full diagnostic
"""
import sys
import time
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

# Capture flags from the real argv before we override it for the module import
GO       = "--go" in sys.argv
N_FRAMES = 250   # Long enough for the per-particle damping rate at N=1000.
                 # is 10x the paper's N=120 baseline (α = damping/m), so terminal
                 # velocity is ~2 m/s and the duck takes ~140 frames to reach the
                 # floor from spawn y=5.5.  120 frames is pure free-fall, no contact.

# Force the duck scene before importing the module (CLI is parsed at import)
sys.argv = ["gaussian_mesh_poc.py", "--solver", "tl_apic",
            "--n-per-entity", "1000", "--shape", "duck",
            "--material", "1"]
import gaussian_mesh_poc as G


# ============================================================
# Step 2: Tag particles by region on REST positions
# ============================================================

def in_ellipsoid(pts, centre, radii):
    d = (pts - centre) / radii
    return (d * d).sum(axis=1) <= 1.0

rest = G.REST_POS
N    = rest.shape[0]

body_m = in_ellipsoid(rest, np.array([0.0,  0.0,   0.0]),  np.array([1.0, 0.65, 0.85]))
head_m = in_ellipsoid(rest, np.array([0.0,  0.95,  0.25]), np.array([0.45, 0.45, 0.45]))
bill_m = in_ellipsoid(rest, np.array([0.0,  0.80,  0.72]), np.array([0.22, 0.12, 0.28]))
tail_m = in_ellipsoid(rest, np.array([0.0,  0.15, -0.88]), np.array([0.28, 0.30, 0.22]))

# Strict tagging:  head/body/neck/other.  "other" means bill or tail only.
tag = np.full(N, "other", dtype=object)
strict_head = head_m & ~body_m
strict_body = body_m & ~head_m
neck        = head_m &  body_m
in_only_bt  = (bill_m | tail_m) & ~head_m & ~body_m

tag[strict_head] = "head"
tag[strict_body] = "body"
tag[neck]        = "neck"
tag[in_only_bt]  = "other"

counts = {r: int((tag == r).sum()) for r in ("head", "body", "neck", "other")}

print("=" * 64)
print(f"DUCK DIAGNOSTIC  -  rest-pose region tagging  (N = {N})")
print("=" * 64)
print(f"  strict-head : {counts['head']:4d}  ({counts['head']/N*100:.1f}%)")
print(f"  strict-body : {counts['body']:4d}  ({counts['body']/N*100:.1f}%)")
print(f"  neck overlap: {counts['neck']:4d}  ({counts['neck']/N*100:.1f}%)")
print(f"  bill/tail   : {counts['other']:4d}  ({counts['other']/N*100:.1f}%)")
print(f"  total tagged: {sum(counts.values())}  (should equal {N})")
print()

TAG_NAMES = ["head", "body", "neck", "other"]
for name in TAG_NAMES:
    m = tag == name
    if m.any():
        c = rest[m].mean(axis=0)
        print(f"  {name:5s}  rest centroid: ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})   "
              f"y-extent [{rest[m,1].min():+.3f}, {rest[m,1].max():+.3f}]")

if counts["head"] > 0 and counts["body"] > 0:
    c_h = rest[tag == "head"].mean(axis=0)
    c_b = rest[tag == "body"].mean(axis=0)
    d0  = float(np.linalg.norm(c_h - c_b))
    print(f"\nRest centroid separation |head - body| = d0 = {d0:.4f} m")

print()
print(f"Sanity:  expected ~head 130, body ~850, neck small (overlap)")
print(f"         actual    head {counts['head']}, body {counts['body']}, neck {counts['neck']}")

if not GO:
    print("\nStep 2 PAUSED.  Re-run with `--go` to proceed to Steps 3 and 4.")
    sys.exit(0)


# ============================================================
# Step 3: Run the duck drop with per-region logging
# ============================================================

print()
print("=" * 64)
print(f"STEP 3  -  running duck drop for {N_FRAMES} frames")
print("=" * 64)

sim = G.GaussianBallSim(material_key="1")  # rubber
sim.drop()

# JIT warmup so the first measured frame isn't dominated by compile
for _ in range(3):
    sim.step()
print("  (JIT warmed)")

results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
csv_path    = results_dir / f"duck_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

COLS = ["frame", "region", "n_particles",
        "J_mean", "J_min", "J_max",
        "Uyy_mean", "Uyy_min", "Uyy_max",
        "centroid_x", "centroid_y", "centroid_z",
        "velocity_mean_y"]

masks = {name: (tag == name) for name in TAG_NAMES}
masks["all"] = np.ones(N, dtype=bool)
REGIONS = TAG_NAMES + ["all"]

def region_row(frame, name, mask, positions, vel_y, Fs, Us):
    if not mask.any():
        return None
    p, v, F, U = positions[mask], vel_y[mask], Fs[mask], Us[mask]
    J   = np.linalg.det(F)
    Uyy = U[:, 1, 1]
    return [
        frame, name, int(mask.sum()),
        float(J.mean()),   float(J.min()),   float(J.max()),
        float(Uyy.mean()), float(Uyy.min()), float(Uyy.max()),
        float(p[:,0].mean()), float(p[:,1].mean()), float(p[:,2].mean()),
        float(v.mean()),
    ]

t_start = time.perf_counter()
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(COLS)
    for frame in range(N_FRAMES):
        sim.step()
        pos, vel, Fs, Us = sim.positions, sim.velocities, sim.Fs, sim.Us
        for r in REGIONS:
            row = region_row(frame, r, masks[r], pos, vel[:, 1], Fs, Us)
            if row is not None:
                w.writerow(row)
        if frame % 20 == 0:
            print(f"  frame {frame:4d}/{N_FRAMES}  "
                  f"all y={pos[:,1].mean():+.3f}  "
                  f"head y={pos[masks['head'],1].mean():+.3f}  "
                  f"body y={pos[masks['body'],1].mean():+.3f}")
print(f"\n  Done in {time.perf_counter()-t_start:.1f}s.  CSV at {csv_path}")


# ============================================================
# Step 4: Analyse and report
# ============================================================

print()
print("=" * 64)
print("STEP 4  -  analysis")
print("=" * 64)

rows = list(csv.DictReader(open(csv_path)))
def col(region, key):
    return np.array([float(r[key]) for r in rows if r["region"] == region])

head_c = np.stack([col("head", "centroid_x"), col("head", "centroid_y"), col("head", "centroid_z")], axis=1)
body_c = np.stack([col("body", "centroid_x"), col("body", "centroid_y"), col("body", "centroid_z")], axis=1)
dist   = np.linalg.norm(head_c - body_c, axis=1)
all_y  = col("all", "centroid_y")
peak_y_frame = int(np.argmin(all_y))   # frame at peak compression (ball lowest)

print(f"\n1. HEAD-BODY CENTROID DISTANCE")
d_rest = dist[0]
print(f"   Frame 0 (rest)        : {d_rest:.4f} m")
print(f"   Frame {peak_y_frame:3d} (peak impact): {dist[peak_y_frame]:.4f} m   (ball y={all_y[peak_y_frame]:+.3f})")
print(f"   Frame {N_FRAMES-1:3d} (final)      : {dist[-1]:.4f} m")
print(f"   Min over run          : {dist.min():.4f} m  at frame {int(dist.argmin())}")
print(f"   Max over run          : {dist.max():.4f} m  at frame {int(dist.argmax())}")
growth_pct = (dist.max() - d_rest) / d_rest * 100
shrink_pct = (dist.min() - d_rest) / d_rest * 100
print(f"   Max swelling from d0  : {growth_pct:+.1f}%")
print(f"   Max compression from d0: {shrink_pct:+.1f}%")

print(f"\n2. PER-REGION J  (= det F)")
head_J = col("head", "J_mean"); body_J = col("body", "J_mean")
print(f"   head J_mean: rest={head_J[0]:.4f}  min={head_J.min():.4f}  max={head_J.max():.4f}  final={head_J[-1]:.4f}")
print(f"   body J_mean: rest={body_J[0]:.4f}  min={body_J.min():.4f}  max={body_J.max():.4f}  final={body_J[-1]:.4f}")
print(f"   At peak impact (frame {peak_y_frame}):  head J={head_J[peak_y_frame]:.4f}   body J={body_J[peak_y_frame]:.4f}")
j_gap_peak = float(head_J[peak_y_frame] - body_J[peak_y_frame])
print(f"   head - body at peak    : {j_gap_peak:+.4f}   (positive = head less compressed)")

print(f"\n3. PER-REGION U_yy (vertical stretch)")
head_Uyy = col("head", "Uyy_mean"); body_Uyy = col("body", "Uyy_mean")
print(f"   head Uyy_mean: rest={head_Uyy[0]:.4f}  min={head_Uyy.min():.4f}  max={head_Uyy.max():.4f}")
print(f"   body Uyy_mean: rest={body_Uyy[0]:.4f}  min={body_Uyy.min():.4f}  max={body_Uyy.max():.4f}")
print(f"   At peak impact (frame {peak_y_frame}):  head Uyy={head_Uyy[peak_y_frame]:.4f}   body Uyy={body_Uyy[peak_y_frame]:.4f}")

print(f"\n4. SVD CLAMP ACTIVITY  (sigma in [0.2, 5.0])")
# Heuristic: count frames where region's J_min falls below 0.04 (extreme compression
# near the sigma=0.2 floor) or J_max exceeds 50 (extreme stretch near sigma=5).
for r in ("head", "body", "neck", "all"):
    jmin = col(r, "J_min"); jmax = col(r, "J_max")
    nlow  = int((jmin < 0.04).sum())
    nhigh = int((jmax > 50).sum())
    print(f"   {r:5s}: {nlow:3d}/{N_FRAMES} frames near J_low clamp,  {nhigh:3d}/{N_FRAMES} frames near J_high clamp")

print(f"\n5. VERDICT")
print(f"   centroid-distance max growth from d0 : {growth_pct:+.1f}%")
print(f"   J gap at peak impact (head - body)   : {j_gap_peak:+.4f}")

if growth_pct > 20:
    verdict = "(c) COHESION FAILURE"
    reason  = (f"head and body separate by {growth_pct:.0f}% from rest distance  -  the substrate is losing "
               "structural integrity, particles in the two regions are drifting apart.")
elif abs(j_gap_peak) < 0.03:
    verdict = "(b) MESH ARTIFACT"
    reason  = (f"head and body J values track within {abs(j_gap_peak):.3f} of each other through peak impact, "
               "so particles are deforming uniformly across regions; the visible head-vs-body distinction "
               "comes from the IDW skin reading anatomy out of an essentially uniform particle cloud.")
elif j_gap_peak > 0.10:
    verdict = "(a)-LIKE REAL PHYSICS"
    reason  = (f"head genuinely compresses less than body (gap {j_gap_peak:+.3f} at peak). Since density "
               "and material are identical, the asymmetry comes from contact geometry: only body particles "
               "are in the floor contact zone  -  the head sits high enough to never touch the floor, so its "
               "particles experience only inertial transport.")
else:
    verdict = "MIXED  -  partly real, partly mesh"
    reason  = (f"small but real compression contrast (J gap {j_gap_peak:+.3f}); the head genuinely is less "
               "compressed because it sits higher than the contact patch, but the visual effect is amplified "
               "by the mesh skin tracking that asymmetry.")
print(f"\n   --> {verdict}")
print(f"       {reason}")
print(f"\nCSV: {csv_path}")
