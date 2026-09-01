"""Multi-seed comparison: all three IQL parameter-sharing variants on K_{2,2}.

A single training run's outcome is not evidence of which parameter-sharing
design "works better" -- it's one draw from whatever distribution of
outcomes that design produces (see the "expect inconsistent convergence"
discussion from earlier: K_{2,2} requires independently-initialised,
independently-exploring signallers to spontaneously specialise, and that
doesn't happen the same way on every seed). This script runs all three
variants across the *same* set of seeds and reports the distribution of
outcomes for each, which is the only way to say anything about the designs
themselves rather than about a handful of particular runs -- this is the
full-scale version of the same logic behind the proposal's own P = 30
independent-runs protocol (Section 4.6.4): NUM_SEEDS below is 30, matching
Table 4.1's methodology rather than the smaller 10-seed pilot this script
started as.

All three variants are trained with *identical* hyperparameters (same
episode count, same epsilon schedule, same seed list) so any difference in
outcomes is attributable to the parameter-sharing design, not to mismatched
settings:

* shared-brain  -- IQLSignaller/IQLGuesser (one network, both roles, masked)
* two-brain     -- TwoBrainIQLSignaller/TwoBrainIQLGuesser (one per role)
* independent   -- IndependentIQLSignaller/IndependentIQLGuesser (one per agent)

"Convergence" here uses the same >= 0.90 mean-reward threshold Section
4.6.3 uses for behavioural/performance identifiability.

This is a genuinely long run (~30 seeds x 3 variants x 4000 episodes) --
expect on the order of an hour. Progress prints per seed, per variant, as
it goes, so partial output is available immediately even if you check in
mid-run.

Run:  python experiments/compare_iql_variants_k22.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.metrics import evaluate
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

NUM_SEEDS = 30
# SEED_START lets a rerun use a genuinely fresh, independent batch of seeds
# rather than replaying the same ones. Note this matters: every seed here
# deterministically controls both torch's weight initialisation and the
# environment's episode sampling (see train_shared_iql()'s
# `torch.manual_seed(seed)` and GraphSignallingParallelEnv's `seed=seed`),
# so rerunning with the *same* seed range would reproduce bit-identical
# results -- that verifies the code is reproducible, not that a result
# holds up. A different seed range is what actually tests robustness.
SEED_START = 30
NUM_EPISODES = 4000
EPSILON_DECAY_EPISODES = 2500
EVAL_EPISODES = 1000
CONVERGENCE_THRESHOLD = 0.90  # Section 4.6.3's behavioural/performance criterion
EVAL_SEED_OFFSET = 10_000  # keeps evaluation's random stream distinct from training's


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)
    return GraphSignallingParallelEnv(graph, seed=seed)


def _evaluate_greedy(signaller_agents, guesser_agents, seed: int) -> float:
    for agent in list(signaller_agents.values()) + list(guesser_agents.values()):
        agent.epsilon = 0.0
    eval_env = make_env(seed=seed + EVAL_SEED_OFFSET)
    result = evaluate(
        eval_env, signaller_agents, guesser_agents,
        episodes=EVAL_EPISODES, seed=seed + EVAL_SEED_OFFSET,
    )
    return result.mean_reward


def run_shared_brain(seed: int) -> float:
    env = make_env(seed=seed)
    _network, signaller_agents, guesser_agents, _history = train_shared_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )
    return _evaluate_greedy(signaller_agents, guesser_agents, seed)


def run_two_brain(seed: int) -> float:
    env = make_env(seed=seed)
    _sig_net, _gue_net, signaller_agents, guesser_agents, _history = train_two_brain_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )
    return _evaluate_greedy(signaller_agents, guesser_agents, seed)


def run_independent(seed: int) -> float:
    env = make_env(seed=seed)
    _sig_nets, _gue_nets, signaller_agents, guesser_agents, _history = train_independent_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )
    return _evaluate_greedy(signaller_agents, guesser_agents, seed)


def summarize(name: str, rewards: list[float]) -> None:
    n = len(rewards)
    mean_reward = sum(rewards) / n
    converged = sum(1 for r in rewards if r >= CONVERGENCE_THRESHOLD)
    print(f"\n{name}")
    print(f"  per-seed final reward: {[round(r, 3) for r in rewards]}")
    print(f"  mean across seeds:     {mean_reward:.3f}")
    print(f"  convergence rate (>= {CONVERGENCE_THRESHOLD}): {converged}/{n}")


def main() -> None:
    print("=" * 64)
    print("Multi-seed comparison: shared-brain vs. two-brain vs. independent")
    print("IQL on K_{2,2}")
    print(f"{NUM_SEEDS} seeds each (seeds {SEED_START}-{SEED_START + NUM_SEEDS - 1}), "
          f"{NUM_EPISODES} training episodes, {EVAL_EPISODES} eval episodes per seed")
    print("=" * 64)

    shared_rewards: list[float] = []
    two_brain_rewards: list[float] = []
    independent_rewards: list[float] = []
    start = time.time()

    for offset in range(NUM_SEEDS):
        seed = SEED_START + offset
        print(f"\n--- seed {seed} ---", flush=True)

        r_shared = run_shared_brain(seed)
        shared_rewards.append(r_shared)
        print(f"  shared-brain final reward: {r_shared:.3f}", flush=True)

        r_two = run_two_brain(seed)
        two_brain_rewards.append(r_two)
        print(f"  two-brain    final reward: {r_two:.3f}", flush=True)

        r_indep = run_independent(seed)
        independent_rewards.append(r_indep)
        print(f"  independent  final reward: {r_indep:.3f}", flush=True)

        elapsed = time.time() - start
        done = offset + 1
        eta = elapsed / done * (NUM_SEEDS - done)
        print(f"  ({done}/{NUM_SEEDS} seeds done, elapsed {elapsed:.0f}s, "
              f"eta {eta:.0f}s)", flush=True)

    print(f"\n(total wall time: {time.time() - start:.0f}s)")

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    summarize("Shared-brain IQL (one network, both roles, masked)", shared_rewards)
    summarize("Two-brain IQL (separate signaller/guesser networks)", two_brain_rewards)
    summarize("Independent IQL (one network per agent)", independent_rewards)


if __name__ == "__main__":
    main()
