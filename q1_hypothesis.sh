#!/bin/bash
#SBATCH --job-name=gsg-q1-hypothesis
#SBATCH --output=gsg_q1_hypothesis_%j.log
#SBATCH --error=gsg_q1_hypothesis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=2-00:00:00
#
# Question 1 / Objective 4 hypothesis tests on K_{2,2}.
#
# Runs experiments/q1_hypothesis_tests_k22.py: for each IQL variant (shared-brain,
# two-brain, independent) trains N independently-seeded populations (default
# N = 100) under the standard protocol and tests, with 95% confidence intervals
# and Holm-corrected p-values:
#   Q-A  does training always produce the SAME convention? (dominant-convention
#        frequency, uniform-over-24 goodness of fit, entropy vs uniform null,
#        tracking-vs-XOR split vs 1/3, symmetry-pair tests)
#   Q-B  HOW MANY distinct conventions? (observed richness, Chao1, Good-Turing
#        coverage, accumulation curve)
#   Q-C  is the SUCCESS RATE the same every time? (Wilson/Clopper-Pearson CIs,
#        homogeneity across independent seed blocks, reward bands, cross-variant
#        Fisher tests)
# plus determinism (same seed -> bit-identical weights) and stability
# (convention unchanged from episode 4000 to 6000) checks.
#
# The log records every run (reward, band, convention id + catalogue number,
# per-signaller rule, exact 4-world rule table, info metrics, lock-in episode,
# convention trajectory) followed by the full statistical analysis and a
# final "answers" section.
#
# Seeds are trained in parallel, one per CPU (--cpus-per-task above). Override
# with environment variables, e.g.:
#   GSG_NUM_SEEDS=200 sbatch q1_hypothesis.sh
#   GSG_NUM_SEEDS=100 GSG_SEED_START=1000 sbatch q1_hypothesis.sh   (fresh, disjoint seeds)
#
# Works both under SLURM (`sbatch q1_hypothesis.sh`) and as a plain script
# (`bash q1_hypothesis.sh`). Submit it FROM the repo root.
#
# Outputs:
#   results/question1/logs/q1_hypothesis_<timestamp>.log       (tee'd stdout)
#   results/question1/hypothesis/runs.csv                      (one row per run)
#   results/question1/hypothesis/summary.csv                   (every metric, CI, p-value)
#   results/question1/figures/hypothesis/*.png                 (5 figures)
#   gsg_q1_hypothesis_<jobid>.log                              (SLURM only)

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/q1_hypothesis_tests_k22.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/q1_hypothesis_tests_k22.py found there)." >&2
    echo "       Submit q1_hypothesis.sh from the repository root:  cd <repo> && sbatch q1_hypothesis.sh" >&2
    exit 1
fi

mkdir -p results/question1/logs results/question1/hypothesis results/question1/figures/hypothesis

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

# one worker process per allocated CPU; each worker uses a single torch thread
export GSG_WORKERS="${GSG_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="results/question1/logs/q1_hypothesis_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Workers:      $GSG_WORKERS"
echo "Seeds:        ${GSG_NUM_SEEDS:-100} per variant, starting at ${GSG_SEED_START:-0}"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/q1_hypothesis_tests_k22.py 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
