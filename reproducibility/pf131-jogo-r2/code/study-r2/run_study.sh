#!/usr/bin/env bash
# Only the requested LJ75 and LJ102, with native --refine-all. One solve at a time; parallel refinement inside.
set -Ee -o pipefail
trap 'rc=$?; printf "ERREUR_CAMPAGNE rc=%s ligne=%s commande=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2' ERR
KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf_cpp_110
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
MODE="${1:-run}"
if (( $# > 0 )); then shift; fi
OUT="${OUT:-/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2}"
# Refuse timing overlap with the previous top5 campaign; no automatic killing.
if [[ "$MODE" = run ]]; then
  python3 "$KIT/stop_campaign.py" \
    --out /data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ98_R1 --check
fi
exec python3 "$KIT/study.py" "$MODE" --out "$OUT" "$@"
