# Evidence bundles

This directory contains the compact, reviewable half of each performance-evidence bundle. Every run directory uses `raggedroute.evidence_bundle.v2` and contains:

- `manifest.json`: source revision and clean state, hardware/toolchain, semantic contract, exact commands when historically available, validation state, dispatch decision, and a complete `git|release|unavailable` file inventory;
- `SHA256SUMS`: checksums for every Git-resident file in that run directory;
- reports, comparisons, compact CSV summaries, and normalized profiler metrics.

Raw JSONL, full aggregate JSON, benchmark/profiler run manifests, `.ncu-rep`, `.nsys-rep`, and SQLite files are stored in the immutable `<run-id>-raw.zip` asset named by the manifest. A missing historical source is never silently omitted: it remains listed with `storage: unavailable` and a reason. `unavailable` lowers the reproducibility level of only the affected claim.

The planned asset tag is `evidence-2026-08-03-v1`. The tag and assets must be created from the PR 3 merge commit; URLs in the manifests are intentionally fixed to that tag. Do not publish the assets from an unmerged feature branch.

Validate Git-resident evidence:

```powershell
python scripts/validate_evidence.py
```

After downloading all release assets into one directory, validate both Git files and archives:

```powershell
python scripts/validate_evidence.py --archives path/to/downloaded/assets
```

The packager refuses to overwrite an existing archive or an already migrated v2 bundle. Historical migration used SHA-matched recovery roots; the local paths themselves are not written to the public v2 manifests.
