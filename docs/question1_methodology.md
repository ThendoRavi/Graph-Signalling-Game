# Question 1 — Methodology to Code Map

Question 1 (Section 4.6) characterises the conventions that emerge on the fixed
complete bipartite graph **K_{2,2}** (two signallers, two guessers) under a
single binary signalling constraint.

## Environment (Section 4.2)
- `src/gsg/environment/graph_signalling_game.py` — one-step Dec-POMDP, tuple
  `G = <N, x, {O_i}, {A_i}, R, G>` (Eq. 4.1); team reward = mean correct
  guesses (Eq. 4.4).
- `src/gsg/environment/graphs.py` — K_{2,2} construction; neighbourhoods.
- `src/gsg/environment/spaces.py` — observation/action spaces; binary and
  ternary (Sec 4.6.6) item spaces.

## Agents & learning (Sections 4.3, 4.4)
- `src/gsg/agents/` — role-specific signaller/guesser; feedforward Q-network
  (2x64, ReLU); replay buffer.
- `src/gsg/algorithms/iql.py` — primary algorithm (R1).
- `src/gsg/algorithms/qmix.py` — CTDE alternative.

## Baselines (Section 4.5)
| Baseline | Module | Tests claim |
|---|---|---|
| Random (R=0.50) | `baselines/random_baseline.py` | learning occurred (i) |
| No-comm (R=0.50) | `baselines/no_comm_baseline.py` | channel usefulness (i) |
| Noisy channel | `baselines/noisy_channel_baseline.py` | robustness (ii) |
| Oracle (R=1.00) | `baselines/oracle_baseline.py` | compression ceiling (iii) |
| Cross-play | `evaluation/cross_play.py` | shareability |

## Protocol (Section 4.6.4)
- P = 30 independent runs, 5e5 episodes each; hyperparameters in Table 4.1
  (`configs/default.yaml`).
- `src/gsg/training/run_manager.py` orchestrates seeds; K = 10 populations for
  cross-play.

## Evaluation criteria (Section 4.6.5)
1. Convergence rate — `evaluation/convergence.py`
2. Mean team reward — `evaluation/metrics.py`
3. Mutual information I(M_i; x_j) — `evaluation/information_theory.py`
4. Channel utilisation H(M_i)/1 — `evaluation/information_theory.py`
5. Entropy reduction — `evaluation/information_theory.py`
6. Joint convention distribution — `evaluation/convention_distribution.py`
7. Complementarity rate — `evaluation/convention_distribution.py`
8. Convention entropy (Eq. 4.14) — `evaluation/convention_distribution.py`
9. Cross-play gap (Eq.) — `evaluation/cross_play.py`

Convention extraction/enumeration: `evaluation/conventions.py`.

## Extended variant (Section 4.6.6)
Ternary items on K_{2,2}, primary IQL only —
`experiments/question1/run_extended_variant.py`,
`configs/question1/k22_ternary_iql.yaml`.
