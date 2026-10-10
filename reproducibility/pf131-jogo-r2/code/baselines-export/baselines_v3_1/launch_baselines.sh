#!/usr/bin/env bash
# Run as a script, never source into an interactive shell.
set -u
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 2
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHON="${PYTHON:-python3}"
EXPLICIT_BACKEND="${BACKEND:-}"
BACKEND="${BACKEND:-cpu}"
# Honour the old explicit switch, without retaining v2's implicit GPU default.
if [[ -n "${LJ_USE_GPU:-}" ]]; then
  if [[ "$LJ_USE_GPU" == 1 ]]; then LEGACY_BACKEND=cupy
  elif [[ "$LJ_USE_GPU" == 0 ]]; then LEGACY_BACKEND=cpu
  else echo 'ERROR: LJ_USE_GPU must be 0 or 1' >&2; exit 2; fi
  if [[ -n "$EXPLICIT_BACKEND" && "$EXPLICIT_BACKEND" != "$LEGACY_BACKEND" ]]; then
    echo 'ERROR: BACKEND conflicts with LJ_USE_GPU; unset LJ_USE_GPU or make both explicit settings agree' >&2; exit 2
  fi
  BACKEND="$LEGACY_BACKEND"
fi
DEFAULT_JOBS=4
[[ "$BACKEND" == cupy ]] && DEFAULT_JOBS=1
read -r -a ATOM_ARGS <<< "${ATOMS:-38 102 103 104 75 76 77 98}"
read -r -a METHOD_ARGS <<< "${METHODS:-BH-OJ GA}"
EXTRA=()
[[ "${DRY:-0}" == 1 ]] && EXTRA+=(--dry-run)
exec "$PYTHON" "$HERE/campaign.py" \
  --atoms "${ATOM_ARGS[@]}" --methods "${METHOD_ARGS[@]}" \
  --runs "${RUNS:-30}" --jobs "${JOBS:-$DEFAULT_JOBS}" \
  --eval-config "${CONFIG:-$HERE/baseline_eval_limits_v3.json}" \
  --outdir "${OUTDIR:-$HERE/results_v3}" --base-seed "${BASE_SEED:-2024}" \
  --backend "$BACKEND" --job-seconds "${JOB_SECONDS:-36000}" \
  --restart-seconds "${RESTART_SECONDS:-0}" "${EXTRA[@]}" "$@"
