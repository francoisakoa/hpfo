# Reviewer guide

## Read without rerunning experiments
- Inspect provenance/PREPARATION.json and BASELINES_IMPORT.json.
- Download the release assets and verify their SHA256SUMS.txt.
- Extract the assets here: they create data/main and data/study.
- Run:
  python3 tools/rebuild_tables.py --data data --output tables-rebuilt
- Compare tables-rebuilt with tables.

## Inspect implementation and accounting
- Main implementation: code/main-pf131.
- Component-study implementation and driver: code/study-r2.
- Read SOURCE_DIFFERENCES.json before comparing the two.
- Inspect src/local.cpp, src/solver.cpp, src/io.cpp, src/seed.cpp,
  the worker implementation, and the study.py task definitions.
- Original commands, applied options, input populations and completion records
  remain in the data assets.

## Build and replay
Use a new build directory and a new output directory.
Read the original CMakeLists.txt and packaged build instructions.
Typical CMake configuration, to be checked against the archived source:
  cmake -S code/main-pf131 -B _build/main -DCMAKE_BUILD_TYPE=Release     -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPF_NATIVE=ON     -DPF_WITH_IPOPT=ON -DPF_WITH_OPENMP=ON     -DPF_WITH_MUMPS=ON -DPF_WITH_METIS=ON -DBUILD_TESTING=ON
  cmake --build _build/main --parallel 8
  ctest --test-dir _build/main --output-on-failure

Repeat separately for code/study-r2/source. Preserve the study driver's
directory conventions and use its own documented build/launch procedure.
Inspect --help before replaying a driver.

Main-run example, after extracting the data:
  _build/main/pf_solver --atoms 38 --runs 30 --seed 2024     --population data/main/LJ38/mumps/LJ38_run1/initial.pfpop     --max-iter 1000 --no-improve-stop 0 --max-time 36000     --threads 10 --workers 8 --refine-all     --local-max-iter 5000 --local-tol 1e-8 --local-refine ipopt     --newton-linear-solver mumps --newton-linear-ordering metis     --ipopt-linear-solver mumps --include-ipopt --output-dir _reruns/LJ38

This launches experiments and is intentionally not run during preparation.
The production environment snapshot is evidence, not a promise that a newly
resolved environment will reproduce timings or trajectories bit for bit.

Baseline verification must include the exact selected runs, calibration budget
file, implementation snapshot and local-solver settings. A baseline archive
import alone is not a certification of that selection.

Before editorial submission, the author must record a successful clean build,
the actual commit/tag, release availability, and any remaining limitations.
