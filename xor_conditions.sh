#!/bin/bash
#SBATCH --job-name=gsg-xor
#SBATCH --output=gsg_xor_%j.log
#SBATCH --error=gsg_xor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00
#
# What drives XOR (holistic) conventions on K_{2,2}?
#
# Runs experiments/xor_conditions_k22.py, a focused follow-up to the
# convention-diversity run. XOR-type perfect conventions appeared only three
# times there -- all in the independent variant, only when network-init and
# item-order were pinned and the exploration/replay RNG was swept. This job
# tests what actually drives that, in three sub-experiments on K_{2,2}:
#
#   1. Base XOR rate under the standard coupled protocol (independent, large N).
#   2. Exploration-seed sweep at several fixed (init, env) launch points, to
#      separate "the exploration stream reaches XOR" from "a specific item
#      order matters" from "rare chance".
#   3. Architecture control: can shared / two_brain reach XOR at all?
#
# Works both under SLURM (`sbatch xor_conditions.sh`) and as a plain script
# (`bash xor_conditions.sh`) on any SSH box. Submit it FROM the repo root
# (`cd <repo> && sbatch xor_conditions.sh`); under SLURM that submit dir is how
# the repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/logs/xor_conditions_<timestamp>.log  (tee'd stdout)
#   gsg_xor_<jobid>.log                                     (SLURM only)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/xor_conditions_k22.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/xor_conditions_k22.py found there)." >&2
    echo "       Submit xor_conditions.sh from the repository root:  cd <repo> && sbatch xor_conditions.sh" >&2
    exit 1
fi

mkdir -p results/question1/logs

# --- pick a Python interpreter -------------------------------------------
if [ -n "${GSG_PYTHON:-}" ]; then
    PYTHON="$GSG_PYTHON"
elif [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    PYTHON="python"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="results/question1/logs/xor_conditions_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/xor_conditions_k22.py 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
