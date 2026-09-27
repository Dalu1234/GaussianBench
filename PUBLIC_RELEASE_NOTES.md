# Public release notes

This repository is the cleaned public form of the GaussianBench v0.6.0 artifact.

The release cleanup does not modify benchmark formulas, scene parameters, tolerances, validity gates, scores, or reported verdicts. It adds repository licensing and contributor documentation, removes generated test caches, repairs links in lightweight report snapshots, and records compatibility-patch hashes separately from clean upstream revisions.

The GaussianFlesh numerical path is preserved. Its large Taichi orchestration module is documented in `reference_entrant/gaussianflesh/ARCHITECTURE.md` instead of being reorganized after validation.

Release verification:

- GaussianBench scorer tests: 66 passed
- GaussianFlesh lightweight tests: 12 passed
- External-entry artifact audit: passed
- Python syntax and packaged JSON validation: passed
- Local Markdown links: passed
- Credential and local-path scan: passed
