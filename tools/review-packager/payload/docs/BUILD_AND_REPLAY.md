# Build, test and replay

## Dependencies

Use the archived environment records in
`reproducibility/pf131-jogo-r2/provenance/main-environment` and `provenance/study`
for the production package versions. The PF build requires a C++17 compiler,
CMake, Eigen3 headers, OpenMP, IPOPT, sequential MUMPS, METIS, and Python for its
existing integration tests. Python tests may also need NumPy, SciPy and scikit-learn
as specified in the archived source. Baseline Python requirements are preserved
with each baseline source version; do not silently replace one version with another.

An existing environment on the author's TRX is `pf_cpp_110`. From the `hpfo` root:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
python3 scripts/build_and_test.py --jobs 4 --native
```

`--native` requests host-specific compilation, appropriate for a local check on the
TRX. Omit it for a portable ISA build. The script prints the actual CMake and CTest
commands. It creates **separate fresh build directories**, requires IPOPT, MUMPS
and METIS support, and runs the existing tests with a 300-second limit per test.
It executes unit/integration test problems, **not the full benchmark campaigns**.

For example, the configuration step is:

```bash
cmake -S reproducibility/pf131-jogo-r2/code/main-pf131 -B .local/manual-main -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DPF_WITH_IPOPT=ON -DPF_WITH_OPENMP=ON -DPF_WITH_MUMPS=ON -DPF_WITH_METIS=ON -DPF_NATIVE=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build .local/manual-main --parallel 4
ctest --test-dir .local/manual-main --output-on-failure --no-tests=error --parallel 1 --timeout 300
```

Use the wrapper for recorded build evidence and a sealed manifest suitable for the
publication checker. The wrapper does not claim tests passed when no tests ran.
Build failures are recorded as failures, not silently bypassed.

## Inspect / replay the archived experiments

First extract the Release assets with `python3 scripts/unpack_data.py`.
From the root, these commands print but do not execute:

```bash
python3 scripts/replay.py main --atoms 38 --runs 30 --output .local/replay-main-LJ38
python3 scripts/replay.py study --configuration full --atoms 75 --run 1 --output .local/replay-study-LJ75-1
python3 scripts/replay.py baseline --method BH-OJ --atoms 38 --run 1 --output .local/replay-baseline-BHOJ-LJ38-1
```

Add `--execute` only to explicitly start that replay. Baseline replay requires the
correct Python environment and a unique original command in the record. It refuses
ambiguous archived source paths. Main parameters are recovered from the archived
options; the main population is checked identical over the selected runs. No reduced
evaluation or time budget is called a reproduction of the production experiment.

The original full-study launcher and CLI documentation remain in `code/study-r2`.
Do not invoke `run_study.sh` unchanged without reviewing its historical absolute
paths: its defaults refer to the original TRX campaign. The convenience replay
wrapper uses new paths instead of touching those originals.

## Diagnostics from stored traces

```bash
python3 scripts/plot_history.py data/main/LJ38/mumps/LJ38_run1/pf_iterations.csv --output .local/diagnostic-LJ38
```

Matplotlib is needed for this optional command. The plots show stored best energy
against iterations, evaluations and seconds, plus stored ESS. They are not labeled
as newly reconstructed versions of the original manuscript figures.
