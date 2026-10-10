# Reviewer guide

## What to inspect

1. **Implementation:** `reproducibility/pf131-jogo-r2/code/main-pf131` and
   `code/study-r2/source`. Read `provenance/SOURCE_DIFFERENCES.json` before comparing
   them. The component-study task driver is `code/study-r2/study.py`.
2. **External baselines:** `code/baselines-export` preserves source and configuration
   paths from the compact archive, including separately frozen versions where present.
   `provenance/BASELINES_IMPORT.json` lists the identified drivers and evaluation
   budget files. No newer implementation is silently substituted.
3. **Evaluation counting:** inspect PF `src/solver.cpp`, `src/local.cpp`,
   `src/ipopt.cpp`, `src/io.cpp`, worker code, and the baseline counting wrapper in
   the archived source. Read `EVALUATION_ACCOUNTING.md` for the timing/counting scope.
4. **Parameters and seeds:** each raw PF JSON contains applied options. Study
   `PLAN.json`, `DONE.json`, and `EXECUTION.json` preserve task selection, hashes and
   original commands. Main populations are frozen by size, not independently redrawn
   for every search seed. Refer to the initialization metadata.
5. **Results:** Tables 7–10 and the run-level indices have exact links and hashes to
   their raw records. Rebuild with `scripts/rebuild_all.py` after extracting the assets.

## Data verification is not an optimization campaign

`verify_package.py` checks delivered source/document hashes. `--assets` additionally
checks archive hashes. `unpack_data.py` checks extracted result hashes. `rebuild_all.py`
rechecks the records and computes summaries; it does not rerun PF, IPOPT or a baseline.
Success fields are not reinterpreted as local-solver convergence.

## Source and binary provenance

The historical main archive SHA256 is recorded in the preserved upstream preparation.
Retrospective collection does not alone certify the historical binary-to-run link.
The old `source_sha256` JSON field is not a hash of the C++ source tree. Separate
source inventories and the study execution records clarify these scopes.

Fresh build evidence is under `validation/build/` once the author or reviewer has
run the build/test script. A local preparation with no such report is not presented
as a successfully compiled/replayed artifact. Actual timings and full trajectories
may vary with compiler, BLAS, IPOPT, hardware and concurrent load.

## Source entry points for replays

Use `scripts/replay.py` to print commands first. Main replay includes the companion
IPOPT multistart. The unchanged main CLI cannot select arbitrary run indices:
`--runs K` means runs 1 through K. Study replay can select a recorded individual task
using its actual `EXECUTION.json`; only local executable/input/output paths change.
Baseline replay uses a unique saved original command where the record contains it;
otherwise the tool refuses to invent it and the archived driver/config remains the
reference. Complete source drivers are included independently of this convenience tool.

## Access

A prepared directory on the TRX is not an accessible repository. The author must
publish the code branch/tag and its Release assets. A GitHub draft Release is not
an adequate data link for ordinary read-only reviewers. For a private repository,
the Editor/reviewers require repository access; knowing the URL is not enough.
The account and repository names identify the author, so this is not anonymous.

## Packaging-tool tests

The assembler and its synthetic tests are retained in `tools/review-packager`.
They do not contain or produce publication benchmark results. They can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/review-packager/tests/test_packaging.py
```

The tests exercise hash checking, archive extraction, preserving original files,
refusing overwrite, the 1200-record grid and dry-run publication behavior.
