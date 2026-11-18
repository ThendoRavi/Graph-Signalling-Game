#!/bin/bash
#SBATCH --job-name=gsg-q2-family-a
#SBATCH --output=gsg_q2_family_a_%j.log
#SBATCH --error=gsg_q2_family_a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bigbatch
#SBATCH --time=2-00:00:00
#
# Question 2, Family A: complete-bipartite ratio sweep, with inverse graphs.
#
# Runs experiments/q2_family_a.py on twelve graphs (all three IQL variants, every seed):
#   A3 K1,4   A2 K1,3   A1 K1,2   A6 K2,4   A5 K2,3   A4 K2,2
#   A7 K3,3   A8 K3,2   A1r K2,1  A6r K4,2  A9 K3,1   A3r K4,1
# Inverse pairs: A1/A1r, A2/A9, A3/A3r, A5/A8, A6/A6r. A4 and A7 are their own inverses.
#
# Each graph is compared with random play, the best pure tracking code, and its exact
# oracle ceiling. The log records baselines, the convention space, success rates with
# confidence intervals, conventions and families, imperfect-run dynamics, cross-play,
# policies during training for the first seeds, and every run's trajectory, followed by
# the counterpart and cross-graph comparisons.
#
# RESUMING: every finished run is saved to results/question2/family_a/runs.jsonl. If the
# job hits its time limit, submit the same command again: finished runs are skipped and
# the analysis runs once all of them are done.
#
# Submit FROM the repo root:
#   sbatch q2_family_a.sh
# Options (environment variables):
#   GSG_Q2_SEEDS=100          seeds per (graph, variant)
#   GSG_Q2_CONDITIONS=A2,A9   only these graphs
#   GSG_Q2_STAGE=analyse      redo the analysis and figures from saved runs, no training
#
# Outputs:
#   gsg_q2_family_a_<jobid>.log                     full log (SLURM)
#   results/question2/logs/q2_family_a_<ts>.log     same log (tee)
#   results/question2/family_a/runs.csv, trajectories.csv, summary.csv
#   results/question2/family_a/figures/<condition>/ and figures/summary/

set -euo pipefail

# --- locate the repo root ------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f experiments/q2_family_a.py ]; then
    echo "ERROR: $REPO_ROOT is not the repo root" >&2
    echo "       (no experiments/q2_family_a.py found there)." >&2
    echo "       Submit q2_family_a.sh from the repository root:  cd <repo> && sbatch q2_family_a.sh" >&2
    exit 1
fi

mkdir -p results/question2/logs results/question2/family_a

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
RUN_LOG="results/question2/logs/q2_family_a_${TIMESTAMP}.log"

echo "Started at:   $(date)"
echo "Repo root:    $REPO_ROOT"
echo "Python:       $($PYTHON --version 2>&1) ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "Workers:      $GSG_WORKERS"
echo "Stage:        ${GSG_Q2_STAGE:-all}"
echo "Run log:      $RUN_LOG"
echo

PYTHONUNBUFFERED=1 "$PYTHON" experiments/q2_family_a.py --stage "${GSG_Q2_STAGE:-all}" 2>&1 | tee "$RUN_LOG"

echo
echo "Done at:      $(date)"
echo "Full log:     $REPO_ROOT/$RUN_LOG"
