#!/bin/bash
#SBATCH --job-name=gsg-conventions
#SBATCH --output=gsg_conventions_%j.log
#SBATCH --error=gsg_conventions_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00        # generous; the whole job is on the order of an hour
#
# Convention diversity + information-theoretic / convention-quality metrics,
# K_{2,2}.
#
# Runs experiments/convention_diversity_k22.py, which:
#
#   Phase 1 -- trains all three IQL parameter-sharing variants (shared-brain,
#   two-brain, independent) for P=30 seeds each on K_{2,2} (matching
#   compare_iql_variants_k22.py's protocol), and for each seed extracts the
#   exact joint signaller convention it landed on plus its full
#   information-theoretic profile (channel utilisation, per-edge mutual
#   information, per-guesser entropy reduction) -- computed exactly via
#   exhaustive enumeration, not sampled. Reports the empirical joint
#   convention distribution, convention entropy, dominant-convention
#   frequency, and complementarity rate per variant.
#
#   Phase 2 -- isolates the three independent sources of randomness a
#   training run draws on (network initialisation / torch, exploration and
#   replay-buffer sampling / Python's `random`, and the environment's
#   hidden-item sampling order) and varies each one alone, holding the other
#   two fixed, to determine which one actually decides which of K_{2,2}'s
#   several equally-rewarding, mirror-symmetric conventions a given run
#   finds.
#
# Works both under SLURM (`sbatch conventions.sh`) and as a plain script
# (`bash conventions.sh`) on any SSH box -- the #SBATCH lines above are just
# comments when run without SLURM. Submit it FROM the repo root
# (`cd <repo> && sbatch conventions.sh`); under SLURM that submit dir is how
# the repo is located (the script itself runs from a read-only spool copy).
#
# Outputs:
#   results/question1/logs/convention_diversity_<timestamp>.log  (tee'd stdout)
#   gsg_conventions_<jobid>.log                                  (SLURM only)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/convention_diversity_k22.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/convention_diversity_k22.py found there)." >&2
    echo "       Submit conventions.sh from the repository root:  cd <repo> && sbatch conventions.sh" >&2
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
RUN_LOG="results/question1/logs/convention_diversity_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/convention_diversity_k22.py 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
