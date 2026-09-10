#!/bin/bash
#SBATCH --job-name=gsg-arm-sweep
#SBATCH --output=gsg_arm_sweep_%j.log
#SBATCH --error=gsg_arm_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00        # 1 day is plenty; the sweep is ~1-2h of CPU
#
# Action-response matrix sweep for Question 1.
#
# Runs experiments/action_response_matrix_sweep.py, which for each graph
# K_{3,3}, K_{4,4}, K_{5,5}, K_{6,6} trains all three IQL variants
# (shared-brain, two-brain, independent) plus the random baseline, then
# saves one action-response matrix figure per (graph, variant) and prints a
# full text summary.
#
# Works both under SLURM (`sbatch train.sh`) and as a plain script
# (`bash train.sh`) on any SSH box -- the #SBATCH lines above are just
# comments when run without SLURM. Submit it FROM the repo root
# (`cd <repo> && sbatch train.sh`); under SLURM that submit dir is how the
# repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/figures/action_response_K{m}x{m}_{variant}.png
#   results/question1/logs/action_response_sweep_<timestamp>.log   (tee'd stdout)
#   gsg_arm_sweep_<jobid>.log  (SLURM only, in the submit dir)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
# Under SLURM the script is run from a read-only spool copy, so
# ${BASH_SOURCE[0]} / $0 points there, not at the repo. SLURM_SUBMIT_DIR is
# the directory `sbatch` was called from -- submit this script from the repo
# root (as with the previous train.py setup) and that's the repo root.
# Outside SLURM, fall back to the directory this script lives in.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/action_response_matrix_sweep.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/action_response_matrix_sweep.py found there)." >&2
    echo "       Submit train.sh from the repository root:  cd <repo> && sbatch train.sh" >&2
    exit 1
fi

mkdir -p results/question1/figures results/question1/logs

# --- pick a Python interpreter -------------------------------------------
# The `py` launcher is Windows-only; on a Linux server use python3/python.
if [ -n "${GSG_PYTHON:-}" ]; then
    PYTHON="$GSG_PYTHON"
elif [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Optional: activate a local venv if one exists and none is active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    PYTHON="python"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="results/question1/logs/action_response_sweep_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Run log:      $RUN_LOG"
echo "LETS COOOK IT UPPP"
echo

# Unbuffered so the tee'd log updates live while the job runs.
PYTHONUNBUFFERED=1 "$PYTHON" experiments/action_response_matrix_sweep.py 2>&1 | tee "$RUN_LOG"

echo
echo "WEEEE DONEEEE"
echo "Done at:      $(date)"
echo "Figures:      $REPO_ROOT/results/question1/figures/"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
