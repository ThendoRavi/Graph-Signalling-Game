# Graph Signalling Game

A bandwidth-constrained cooperative multi-agent signalling game. Agents are
arranged on a directed signalling graph: **signallers** observe the hidden items
of neighbouring **guessers** and emit a single binary signal; guessers observe
the signals they receive and must infer their own hidden item. One bit, one step,
one shared reward.

This repository currently scaffolds **Question 1 — Convention Characterisation**
(methodology Section 4.6), conducted on the fixed complete bipartite graph
**K_{2,2}** (two signallers, two guessers).

## Research Question 1

> Under bandwidth-constrained cooperative signalling, restricted to a single
> binary signal, do MARL agents converge on identifiable and consistent
> conventions, and if so, what is the character of those conventions?

## Layout

```
configs/                YAML experiment/baseline configurations
  question1/            K2,2 IQL (R1), QMIX, ternary variant
  baselines/            random, no-comm, noisy-channel, oracle
src/gsg/                core package
  environment/          Graph Signalling Game env, K2,2 graph, spaces (Sec 4.2)
  agents/               signaller, guesser, Q-network, replay buffer (Sec 4.3-4.4)
  algorithms/           IQL (primary), QMIX (Sec 4.4.1)
  baselines/            random, no-comm, noisy-channel, oracle (Sec 4.5)
  training/             single-run trainer + multi-seed manager (Sec 4.6.4)
  evaluation/           convergence, conventions, information theory,
                        convention distribution, cross-play, metrics (Sec 4.6.5)
  utils/                config, seeding, logging
experiments/question1/  runnable entry points (training, baselines, cross-play,
                        extended variant, analysis)
scripts/                pipeline runners (.ps1 / .sh)
tests/                  unit tests
results/question1/      run outputs (runs / checkpoints / logs / figures)
docs/                   methodology notes
```

## Status

The **core environment is implemented and verified** on the base case of one
signaller and one guesser:

* `environment/` — `SignallingGraph` (any topology) + `GraphSignallingGame`
  (three-phase episode, reward per Eq. 4.4, rule enforcement).
* `agents/` — role-specific `Agent` base classes + deterministic policies.
* `baselines/` — random (floor, R≈0.50) and oracle (ceiling, R=1.00).
* `evaluation/metrics.py` — team-reward evaluation + performance-band classifier.
* `visualization.py` — text and matplotlib renderers of the graph/episode.

The learning code (IQL/QMIX), the remaining baselines, the information-theoretic
metrics, and the K_{2,2} experiments are still stubs.

## Running

```bash
python tests/test_environment.py       # verify the environment rules (8 checks)
python experiments/demo_single_pair.py  # run the base case + save a diagram
```

The base game is fully OOP: the same environment/agent code runs unchanged on
larger graphs (e.g. `SignallingGraph.complete_bipartite(2, 2)` for K_{2,2}) —
you grow the game by adding nodes and edges, not by editing the environment.

See `docs/question1_methodology.md` for the mapping from methodology sections to
modules.
