#!/usr/bin/env bash
# New installation only; the R1 run and its files are not modified here.
set -Ee -o pipefail
trap 'rc=$?; printf "ERREUR_PREPARATION rc=%s ligne=%s commande=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2' ERR
KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OLD_OUT="${1:-/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ98_R1}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
python3 "$KIT/reuse_binaries.py" --previous-out "$OLD_OUT"
python3 -m unittest discover -s "$KIT/tests" -p 'test_runner.py' -v
python3 -m unittest discover -s "$KIT/tests" -p 'test_switch.py' -v
printf '\nKIT_REFINE_ALL_PRET=%s\n' "$KIT"
