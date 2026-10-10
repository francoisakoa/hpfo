# HPFO — PF C++ 1.3.1 reviewer artifact

**Manuscript:** *A Hybrid Particle Filtering and Local Optimization Framework for Solving Challenging Lennard-Jones Cluster Problems*  
**Author:** François Bertrand Akoa  
**Journal submission:** JOGO-D-25-00267  
**Repository:** https://github.com/francoisakoa/hpfo  
**Proposed fixed review tag:** `pf131-jogo-r2` (not a statement that it has already been published)

Start with [the reviewer guide](docs/REVIEWER_GUIDE.md).

**PF1.3.1 solver source:** [main-pf131](reproducibility/pf131-jogo-r2/code/main-pf131),
including `src/solver.cpp`, `src/local.cpp`, `app/main.cpp` and `CMakeLists.txt`.
The compiled executable is named `pf_solver`; it is built from this source, not
included as a platform-specific binary in this source artifact.
The separate instrumented study source is
[study-r2/source](reproducibility/pf131-jogo-r2/code/study-r2/source).


This artifact distinguishes the archived main PF131 implementation from the
instrumented component-study implementation. Neither is replaced by PF1.4/PF1.5.
The source snapshots and experiment drivers are under
[`reproducibility/pf131-jogo-r2/code`](reproducibility/pf131-jogo-r2/code).

| Dataset | Distinct results | Interpretation |
|---|---:|---|
| Main PF131 benchmark | 240 | 30 search runs on each of eight LJ sizes |
| Associated IPOPT multistart | 240 | Companion benchmark; do not pool with PF |
| Component study | 820 | 600 ablation runs + 220 additional sensitivity runs |
| External baselines | 1,200 | Five implementations × eight sizes × 30 runs |

**Table 7 uses the component-study `full` control and all 30 repetitions.**
Tables 8–10 use ten repetitions; their shared default references use only
component-study `full` runs 1–10. These are not extra independent simulations.
The main benchmark and the component study are not pooled.

## Quick inspection — no new optimization

```bash
python3 scripts/verify_package.py
```

The large raw data archives are GitHub **Release assets**, not Git-tracked files.
On the author's TRX they are already in `release-assets/`. After cloning the tagged
review artifact, download them from its Release, then:

```bash
bash scripts/fetch_release_assets.sh pf131-jogo-r2
python3 scripts/verify_package.py --assets
python3 scripts/unpack_data.py
python3 scripts/rebuild_all.py --output .local/tables-rebuilt
```

The last command verifies the indexed JSON hashes and rebuilds Tables 7–10 and
main, multistart and external-baseline summaries from raw records. It never calls
an optimizer. CSV summaries are generated machine-readable outputs, not new runs.

## Build / test / explicit replay

See [BUILD_AND_REPLAY.md](docs/BUILD_AND_REPLAY.md). The repository supplies separate
clean builds and their original CTest tests. A successful build does not prove
that an executable generated today is the executable used historically.
Replays are dry runs unless `--execute` is supplied, and require new output paths.
No original campaign result is overwritten.

## Interpretation and limits

Success means attainment of the **recorded energy target** at the recorded
tolerance, not a certificate of global optimality or stationarity. Local work is
included in objective counts. Timings are elapsed solver times, not summed worker
CPU times. The two PF source snapshots, their flow schedules and parameter records
remain separately traceable.

The theoretical statements for an idealized uncapped surrogate are not convergence
guarantees for the practical capped, adaptive, archive-dependent implementation.
Baseline claims concern the specific implementations and adopted comparable-budget
protocol, not entire algorithmic families. Validation is restricted to the stated
Lennard-Jones experiments.

Raw baseline event-by-event logs are not supplied by the compact export; the full
backups remain with the author. `plot_history.py` creates diagnostics from stored
PF traces; these are not claimed to be the exact submitted manuscript figures.
Read [ARTIFACT_SCOPE.md](docs/ARTIFACT_SCOPE.md) for those boundaries.

Existing source/license notices are retained. This packaging does not invent a
software license, grant rights to third-party material, or include proprietary
HSL binaries. See the retained `LICENSE_POLICY.md` and `THIRD_PARTY_NOTICES.md` files.
