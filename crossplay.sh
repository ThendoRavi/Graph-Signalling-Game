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
# Runs experiments/crossplay_k22.py TWICE, once per experiment:
#
#   1. --mode all      the normal experiment: seeds 0..14 per variant, exactly
#                      as trained, so imperfect runs (R = 0.875 / 0.75) are
#                      included and some self-play cells are below 1.0.
#   2. --mode perfect  only runs that achieved perfect coordination: seeds are
#                      scanned from 0 upward and the first 15 whose self-play is
#                      exactly R = 1.0 are kept (the rest are logged and
#                      excluded). Every self-play cell is 1.0, so any cross-play
#                      drop is purely a mismatch between two perfect conventions.
#
# Both modes share the trained-model cache, so experiment 2 reuses seeds 0..14
# from experiment 1 and only trains the extra seeds it needs.
#
# For each IQL variant (shared-brain, two-brain, independent) each experiment
# produces:
#   * a 15x15 population cross-play matrix (team reward; x = signaller run,
#     y = guesser run). The diagonal is SELF-PLAY: run i's own signallers
#     played against run i's own guessers, through exactly the same evaluation
#     as every off-diagonal cell -- nothing is copied from training;
#   * a 2x2 grid of per-edge cross-play matrices (one per signalling edge
#     S_a -> G_b), isolating how tolerant each edge is to a foreign partner;
#   * the cross-play gap Delta = (mean self-play) - (mean off-diagonal).
# Every pairing is played exactly over all four equally likely worlds (no
# sampling noise); self-play is additionally cross-checked on 400 sampled
# episodes.
#
# Each log (saved via tee below) records, for every run: its seed, its
# self-play reward (exact and sampled) and band, its joint convention id, its
# per-signaller rule, and the full enumerated self-play (all four worlds); then
# the distribution of cross-play cell values (perfect mode: against the
# prediction for uniformly drawn tracking conventions), the convention
# equivalence classes, a convention-vs-convention cross-play table, example
# pairings grouped by outcome (COMPATIBLE / ANTI-COMPATIBLE / ORTHOGONAL /
# PARTIAL), and a final per-variant summary table.
#
# Works both under SLURM (`sbatch crossplay.sh`) and as a plain script
# (`bash crossplay.sh`) on any SSH box. Submit it FROM the repo root
# (`cd <repo> && sbatch crossplay.sh`); under SLURM that submit dir is how the
# repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/figures/crossplay/crossplay_population_<variant>.png          (experiment 1)
#   results/question1/figures/crossplay/crossplay_edges_<variant>.png               (experiment 1)
#   results/question1/figures/crossplay/crossplay_population_<variant>_perfect.png  (experiment 2)
#   results/question1/figures/crossplay/crossplay_edges_<variant>_perfect.png       (experiment 2)
#   results/question1/figures/crossplay/*.npy                 (raw matrices + seeds per run)
#   results/question1/logs/crossplay_all_<timestamp>.log      (tee'd stdout, experiment 1)
#   results/question1/logs/crossplay_perfect_<timestamp>.log  (tee'd stdout, experiment 2)
#   gsg_crossplay_<jobid>.log                                 (SLURM only, both experiments)
#
# Override the perfect-mode seed scan limit (default 100) with e.g.
#   GSG_CROSSPLAY_MAX_SEEDS=200 sbatch crossplay.sh

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
LOG_ALL="results/question1/logs/crossplay_all_${TIMESTAMP}.log"
LOG_PERFECT="results/question1/logs/crossplay_perfect_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Logs:         $LOG_ALL"
echo "              $LOG_PERFECT"
echo

echo "=== Experiment 1/2: all runs (seeds 0..14, imperfect included) ==="
PYTHONUNBUFFERED=1 "$PYTHON" experiments/crossplay_k22.py --mode all 2>&1 | tee "$LOG_ALL"

echo
echo "=== Experiment 2/2: perfect runs only (self-play R = 1.0) ==="
PYTHONUNBUFFERED=1 "$PYTHON" experiments/crossplay_k22.py --mode perfect 2>&1 | tee "$LOG_PERFECT"

echo
echo "Done at:      $(date)"
echo "Logs:         $REPO_ROOT/$LOG_ALL"
echo "              $REPO_ROOT/$LOG_PERFECT"
