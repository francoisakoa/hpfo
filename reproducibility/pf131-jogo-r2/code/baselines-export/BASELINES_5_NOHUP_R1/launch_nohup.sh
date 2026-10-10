#!/usr/bin/env bash
# Execute this file with bash; never source it in an interactive shell.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 2
BASE="${BASE:-/data/projects/PF_GlobalOptimization}"
PY="${PYTHON:-$BASE/baselines_v3/.venv/bin/python}"
ROOT="${OUTDIR:-$BASE/results_baselines_nohup_R1}"
if [[ "${TECHNICAL_SMOKE:-0}" == 1 && -z "${OUTDIR:-}" ]]; then ROOT="$BASE/results_baselines_nohup_TECHNICAL"; fi
JOBS="${JOBS:-22}"
if [[ ! -x "$PY" ]]; then printf 'Python missing: %s\n' "$PY" >&2; exit 2; fi
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS LJ_USE_GPU
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$PY" "$HERE/verify_kit.py" || exit $?
ARGS=(--base "$BASE" --root "$ROOT" --python "$PY" --jobs "$JOBS")
if [[ "${ALLOW_SMT:-0}" == 1 ]]; then ARGS+=(--allow-smt); fi
if [[ "${TECHNICAL_SMOKE:-0}" == 1 ]]; then ARGS+=(--technical-smoke); printf "TECHNICAL TEST ONLY: ten 61-evaluation runs, NOT production.\n"; fi
# No existing process is killed by this launcher. Audit can run before handover.
"$PY" "$HERE/supervise.py" plan "${ARGS[@]}" || exit $?
mkdir -p "$ROOT" || exit 2
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
LOG="$ROOT/nohup_${STAMP}.log"
printf 'Starting detached supervisor, global concurrency %s.\nLOG=%s\n' "$JOBS" "$LOG"
nohup "$PY" -u "$HERE/supervise.py" run "${ARGS[@]}" </dev/null >"$LOG" 2>&1 &
PID=$!
# Handshake: a live PID alone is not a successful launch. Preparation can take time.
for ((i=0;i<180;i++)); do
  if grep -q '^READY:' "$LOG"; then
    printf '%s\n' "$PID" > "$ROOT/last_launcher.pid"
    printf '%s\n' "$LOG" > "$ROOT/last_log.txt"
    printf 'NOHUP_READY PID=%s LOG=%s\n' "$PID" "$LOG"
    printf 'All five methods are in the shared queue. Ctrl+C on tail stops tail only.\n'
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"; RC=$?
    tail -n 60 "$LOG"
    printf 'NOHUP_START_FAILED rc=%s; original results untouched.\n' "$RC" >&2
    exit "${RC:-2}"
  fi
  sleep 1
done
printf 'Preparation still running or waiting; PID=%s; inspect %s. No second launch.\n' "$PID" "$LOG"
exit 3
