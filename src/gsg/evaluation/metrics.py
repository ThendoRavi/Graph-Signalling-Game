"""Shared metric helpers and result containers (Sections 4.5.6, 4.6.5).

At this base stage the evaluation we need is deliberately small: run an agent
pair for many episodes and measure how often the guessers are right. This module
provides that, plus the performance-band classifier that gives each mean reward a
qualitative reading (Section 4.5.6). The information-theoretic and
convention-distribution metrics (mutual information, convention entropy,
cross-play gap) get their own modules once training exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..environment.graph_signalling_game import GraphSignallingGame


@dataclass
class EvaluationResult:
    """Aggregate outcome of evaluating an agent pair over many episodes."""

    episodes: int
    mean_reward: float                       # average team reward (Eq. 4.4)
    per_guesser_accuracy: Dict[int, float]   # guesser_id -> fraction correct
    band: str                                # qualitative reading (Section 4.5.6)

    def __str__(self) -> str:
        acc = ", ".join(f"G{g}={a:.3f}" for g, a in sorted(self.per_guesser_accuracy.items()))
        return (
            f"mean_reward={self.mean_reward:.4f} over {self.episodes} eps "
            f"[{self.band}] | per-guesser: {acc}"
        )


def classify_performance(mean_reward: float, floor: float = 0.5, tol: float = 1e-9) -> str:
    """Map a mean team reward to a qualitative band (Section 4.5.6).

    The thresholds (0.50 floor, 0.75 single-bit ceiling, 1.00 perfect) are the
    ones the proposal uses to read what *kind* of outcome a number represents,
    not merely whether it is high. ``floor`` is exposed because the chance floor
    is ``1 / num_item_values`` (0.50 binary, 0.333 ternary). ``tol`` widens each
    exact threshold into a small band so that a Monte-Carlo estimate sitting a
    hair off (e.g. random at 0.498) is read as "at floor" rather than as
    genuine miscommunication.
    """
    if mean_reward < floor - tol:
        return "below floor (active miscommunication)"
    if mean_reward <= floor + tol:
        return "at floor (no convention)"
    if mean_reward < 0.75 - tol:
        return "partial convention"
    if abs(mean_reward - 0.75) <= tol:
        return "single-bit ceiling"
    if mean_reward < 1.0 - tol:
        return "exploiting joint channel"
    return "perfect coordination"


def evaluate(
    env: GraphSignallingGame,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
    episodes: int = 10_000,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """Play ``episodes`` episodes and summarise team reward + per-guesser accuracy.

    A single fixed ``seed`` is used to re-seed the environment once at the start,
    so the whole evaluation is reproducible while items still vary episode to
    episode.
    """
    if seed is not None:
        env.reset(seed=seed)

    guessers = env.graph.guessers
    correct_counts: Dict[int, int] = {g: 0 for g in guessers}
    reward_sum = 0.0

    for _ in range(episodes):
        record = env.run_episode(signaller_agents, guesser_agents)
        reward_sum += record.reward
        for g in guessers:
            if record.correct[g]:
                correct_counts[g] += 1

    mean_reward = reward_sum / episodes
    per_guesser = {g: correct_counts[g] / episodes for g in guessers}
    floor = 1.0 / env.item_space.n
    # ~3 standard errors of a mean-of-Bernoulli estimate: absorbs sampling noise
    # around the exact band thresholds so classification is stable across seeds.
    noise_tol = 3.0 * (0.25 / episodes) ** 0.5
    return EvaluationResult(
        episodes=episodes,
        mean_reward=mean_reward,
        per_guesser_accuracy=per_guesser,
        band=classify_performance(mean_reward, floor=floor, tol=noise_tol),
    )
