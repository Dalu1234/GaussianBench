"""Command-line parser for gaussian_mesh_poc.

Kept separate so tests and tooling can inspect CLI defaults without importing the Taichi simulator.
"""

import argparse


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="GaussianFlesh - TL-APIC vs UL-MLS-MPM")
    p.add_argument("--solver", choices=["tl_apic", "ul_mlsmpm"], default="ul_mlsmpm",
                   help="Solver for single-entity mode (ignored in --scene=compare).  "
                        "UL is the PhysGaussian-equivalent default; TL kept as research path "
                        "for fracture / WLS drift correction.")
    p.add_argument("--scene",  choices=["single", "compare", "collide"], default="single",
                   help="single = one entity; compare = TL + UL side by side; "
                        "collide = two UL bodies dropped to collide.")
    p.add_argument("--n-per-entity", type=int, default=203930,
                   help="Particles per entity.  Use 120 to reproduce paper Table 1.")
    p.add_argument("--frames", type=int, default=None,
                   help="Frame budget; default 800 for single, 3000 for compare.")
    p.add_argument("--shape",  default=None,
                   help="Override default shape (sphere/duck/cuboid/bunny/custom).")
    p.add_argument("--custom-obj", default=None,
                   help="Path to an OBJ mesh used when --shape custom.")
    p.add_argument("--material", default=None,
                   help="Material key (e.g. '1' = Rubber).")
    p.add_argument("--csv-dir", default="results",
                   help="Where to write the per-frame drift CSV.")
    p.add_argument("--headless", action="store_true",
                   help="Render-free mode for long batch runs (not yet supported).")
    p.add_argument("--k-bind", type=int, default=None,
                   help="Nearest-particle bindings per mesh vertex (IDW skin).  Default: all particles per entity.")
    p.add_argument("--no-skin", action="store_true",
                   help="Hide the skinned mesh and reveal the particle cloud (dots coloured by sigma_min).")
    p.add_argument("--hand", action="store_true",
                   help="Enable webcam hand tracker (MediaPipe).  Hand becomes a capsule-skeleton collider.")
    p.add_argument("--fracture", action="store_true",
                   help="Enable continuum damage + WLS bond-break (real fracture, permanent damage).")
    p.add_argument("--ply", default=None,
                   help="Path to a 3DGS .ply file to load as the particle source (overrides --shape).")
    p.add_argument("--gs-render", action="store_true",
                   help="Open a second window showing the diff_gaussian_rasterization view of the substrate.")
    p.add_argument("--gs-only", action="store_true",
                   help="Skip PyVista entirely - only run the gs_render path.  Faster (no VTK overhead).")
    p.add_argument("--video", default=None,
                   help="Record gs_render output to MP4 file instead of displaying it.  e.g. --video out.mp4")
    p.add_argument("--ply-upright", action="store_true",
                   help="Rotate a loaded PLY so its trained-Z axis is world-up (pot-down drop).")
    p.add_argument("--ply-split", default=None,
                   help="Multi-material split for a PLY body: low side of the chosen axis gets "
                        "the specified material.  Format: MATKEY:FRACTION  e.g. --ply-split "
                        "5:0.35 makes the low 35%% Wood while the rest keeps --material.")
    p.add_argument("--ply-split-axis", choices=["x", "y", "z"], default="y",
                   help="Axis used by --ply-split. Default y keeps the previous bottom/top split; "
                        "x is useful for left/right material demos.")
    p.add_argument("--impact-vy", type=float, default=-4.0,
                   help="Initial downward velocity for --ply-split poster collision videos.")
    p.add_argument("--squash-release", action="store_true",
                   help="Headless poster shot: place the PLY on the floor, press it with a top "
                        "plate, hold, then release so elastic/plastic recovery is visible.")
    p.add_argument("--squash-ratio", type=float, default=0.42,
                   help="Target compressed height as a fraction of the initial body height for "
                        "--squash-release. Lower means a stronger squash.")
    p.add_argument("--side-squash-release", action="store_true",
                   help="Headless poster shot: two vertical side plates squeeze along x, hold, "
                        "then move away.")
    p.add_argument("--dual-squash-release", action="store_true",
                   help="Headless poster shot: top and bottom horizontal plates squeeze vertically, "
                        "hold, then both move away.")
    p.add_argument("--drag-demo", action="store_true",
                   help="Headless poster shot: scripted sphere collider first drags the lower/base "
                        "region, then the upper/canopy region, to show per-particle material contrast.")
    p.add_argument("--ficus-drop-compare", action="store_true",
                   help="Headless poster shot: two copies of one PLY side by side. Left is upright "
                        "so metal pot/base hits first; right is inverted so jelly canopy hits first.")
    p.add_argument("--ficus-metal-compare", action="store_true",
                   help="Headless poster shot: two upright copies of one PLY side by side. Left is "
                        "bottom-metal/top-jelly; right is uniform Metal.")
    p.add_argument("--gravity-y", type=float, default=-9.8,
                   help="World y gravity acceleration. Use 0 for isolated squeeze/release demos.")
    p.add_argument("--material-mu", type=float, default=None,
                   help="Override the selected material shear modulus (Pa).")
    p.add_argument("--material-lambda", type=float, default=None,
                   help="Override the selected material first Lame parameter (Pa).")
    p.add_argument("--material-density", type=float, default=None,
                   help="Override the selected material density (kg/m^3).")
    p.add_argument("--material-damping", type=float, default=None,
                   help="Override the selected material damping coefficient.")
    p.add_argument("--substep-dt", type=float, default=None,
                   help="Physics substep size in seconds. Overrides the default frame/substep ratio.")
    p.add_argument("--frame-dt", type=float, default=None,
                   help="Simulation time advanced per rendered frame in seconds.")
    p.add_argument("--grid-dx", type=float, default=None,
                   help="Override MPM grid spacing in world units.")
    p.add_argument("--grid-lim", type=float, default=None,
                   help="Override the UL world-grid side length (PhysGaussian grid_lim).")
    p.add_argument("--grid-v-damping-scale", type=float, default=1.0,
                   help="Per-substep multiplier applied to updated UL grid velocities.")
    p.add_argument("--physgaussian-ficus-match", action="store_true",
                   help="Run the stock PhysGaussian ficus experiment with matched preprocessing, "
                        "volumes, density regions, forcing, and boundary conditions.")
    p.add_argument("--physgaussian-ficus-drop", action="store_true",
                   help="Render the matched full-particle rubber ficus gravity-drop test.")
    p.add_argument("--state-output", default=None,
                   help="Directory for sampled NPZ states in benchmark modes.")
    p.add_argument("--start-bottom", type=float, default=None,
                   help="Requested initial object bottom coordinate in the source Z-up frame.")
    p.add_argument("--floor-coordinate", type=float, default=0.0,
                   help="Requested floor coordinate in the source Z-up frame; used with --start-bottom.")
    p.add_argument("--multi-material", default=None,
                   help="Split the body into two materials by y-coordinate.  Pass the "
                        "material key for the TOP half (e.g. '3' for jelly).  Bottom half "
                        "keeps the primary --material.  Demonstrates per-particle mu.")
    p.add_argument("--grid-fill", action="store_true",
                   help="PhysGaussian-style particle filling: voxelize the shape, fill each "
                        "interior cell with ppc particles, size each particle's V0 + cov to its "
                        "share of the cell.  Result: every particle perfectly covers its share "
                        "of the shape regardless of N.  Without this, particles share a uniform "
                        "V0 = V_shape/N and a fixed cov_spread per material (visual gaps at low N).")
    p.add_argument("--ppc", type=int, default=8,
                   help="Particles per cell for --grid-fill.  Default 8 (PhysGaussian-style).")
    p.add_argument("--record", default=None,
                   help="Auto-start MP4 recording to this path when the viewer opens.")
    p.add_argument("--capture-dir", default="captures",
                   help="Directory for screenshots and toggled MP4 recordings.")
    p.add_argument("--thermo", action="store_true",
                   help="Enable thermodynamics (Stomakhin 2014): per-frame heat diffusion, "
                        "latent heat, and melting (solid-to-fluid phase change driven by temperature).")
    p.add_argument("--hot-floor", type=float, default=None,
                   help="Temperature of the floor heat source for --thermo (Dirichlet BC on the "
                        "floor band).  e.g. --hot-floor 200.  Default: material melt_temp + 60.")
    p.add_argument("--ambient-temp", type=float, default=20.0,
                   help="Initial/ambient temperature for --thermo.  Default 20.")
    p.add_argument("--demo-spread", action="store_true",
                   help="Enable visual floor-spread velocity injection for molten particles "
                        "(--thermo only).  NOT physically derived - for demo use only.  "
                        "Default OFF so the poster run uses only MPM grid dynamics.")
    p.add_argument("--pressure-project", action="store_true",
                   help="Experimental Chorin-style projection for molten fluid nodes. "
                        "Default OFF; current implementation is not validated for poster renders.")
    args, _ = p.parse_known_args(argv)
    return args
