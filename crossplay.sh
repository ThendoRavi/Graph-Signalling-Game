#!/bin/bash
#SBATCH --job-name=gsg-crossplay
#SBATCH --output=gsg_crossplay_%j.log
#SBATCH --error=gsg_crossplay_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00
#
# Cross-play evaluation on K_{2,2} (Section sec:baseline-crossplay).
#
# Runs experiments/crossplay_k22.py, which for each IQL variant (shared-brain,
# two-brain, independent) trains 15 independent populations and measures
# cross-play: pairing signallers from one run with guessers from another.
# Produces, per variant:
#   * a 15x15 population cross-play matrix (mean team reward; x = signaller run,
#     y = guesser run, diagonal = self-play) -- the direct analogue of a Hanabi
#     cross-play matrix, with runs labelled 1..15;
#   * a 2x2 grid of per-edge cross-play matrices (one per signalling edge
#     S_a -> G_b), isolating how tolerant each edge is to a foreign partner;
#   * the cross-play gap Delta = (mean self-play) - (mean off-diagonal).
#
# Works both under SLURM (`sbatch crossplay.sh`) and as a plain script
# (`bash crossplay.sh`) on any SSH box. Submit it FROM the repo root
# (`cd <repo> && sbatch crossplay.sh`); under SLURM that submit dir is how the
# repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/figures/crossplay/crossplay_population_<variant>.png
#   results/question1/figures/crossplay/crossplay_edges_<variant>.png
#   results/question1/figures/crossplay/*.npy                 (raw matrices)
#   results/question1/logs/crossplay_<timestamp>.log          (tee'd stdout)
#   gsg_crossplay_<jobid>.log                                 (SLURM only)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/crossplay_k22.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/crossplay_k22.py found there)." >&2
    echo "       Submit crossplay.sh from the repository root:  cd <repo> && sbatch crossplay.sh" >&2
    exit 1
fi

mkdir -p results/question1/logs results/question1/figures/crossplay

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
RUN_LOG="results/question1/logs/crossplay_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/crossplay_k22.py 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
