#!/bin/bash
#SBATCH --job-name=gsg-conv-kmm
#SBATCH --output=gsg_conv_kmm_%j.log
#SBATCH --error=gsg_conv_kmm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00        # generous; longer schedules at scale
#
# Convention determination on K_{m,m}, m in {3,4,5,6}, with longer training
# and the same mathematical-framework analysis used for K_{2,2}.
#
# Runs experiments/convention_diversity_kmm.py, which for each graph
# K_{3,3}..K_{6,6} trains all three IQL variants (shared-brain, two-brain,
# independent) for a substantially longer episode budget than the earlier
# scaling probe, and for every seed classifies the learned convention
# structurally: each signaller's learned bit is a Boolean function of the m
# observed items, classified as constant (pooling), dictator (a clean
# one-item relay = "tracking"), parity (XOR of several items = holistic), or
# general nonlinear -- the m-item generalisation of the K_{2,2} "track x0 /
# XOR / pooling" catalogue. Reports greedy reward, per-signaller Boolean
# type, per-guesser entropy reduction, the exact joint convention id, mean
# information-theoretic metrics, and 5 example episodes per seed.
#
# Works both under SLURM (`sbatch conventions_kmm.sh`) and as a plain script
# (`bash conventions_kmm.sh`) on any SSH box -- the #SBATCH lines above are
# just comments when run without SLURM. Submit it FROM the repo root
# (`cd <repo> && sbatch conventions_kmm.sh`); under SLURM that submit dir is
# how the repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/logs/convention_kmm_<timestamp>.log  (tee'd stdout)
#   gsg_conv_kmm_<jobid>.log                                (SLURM only)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/convention_diversity_kmm.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/convention_diversity_kmm.py found there)." >&2
    echo "       Submit conventions_kmm.sh from the repository root:  cd <repo> && sbatch conventions_kmm.sh" >&2
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
RUN_LOG="results/question1/logs/convention_kmm_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/convention_diversity_kmm.py 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
