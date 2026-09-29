"""Boolean-function analysis of learned signaller policies.

This is the mathematical framework that generalises the K_{2,2} convention
catalogue (the 24 perfect conventions, 8 "tracking" + 16 "XOR-type") to
K_{m,m}, where the perfect-convention space -- the ``(2^m)!`` bijective
joint encodings -- is far too large to enumerate (already ``8! = 40320`` at
m = 3). Instead of cataloguing the whole space, we classify each *learned*
convention structurally.

A signaller in K_{m,m} observes m binary items and emits one bit, so its
greedy policy is a Boolean function ``f : {0,1}^m -> {0,1}``. Every such
function falls into exactly one of four structural classes, which are the
m-item generalisations of the labels used for K_{2,2}:

* **constant** -- ``f`` ignores every item (a pooling signaller: the light
  is always ON or always OFF, carries no information).
* **dictator** -- ``f`` depends on exactly one item ``x_j`` (a clean one-bit
  relay of that guesser's item, possibly inverted). This is the
  *compositional* / "tracking" case: the light means one guesser's item on
  its own.
* **parity** -- ``f`` is an affine (XOR-linear) function of two or more items,
  ``f(x) = c (+) x_{j1} (+) ... (+) x_{jk}``. This is the *holistic* case:
  the light encodes the parity of several items (the m-item generalisation
  of the K_{2,2} XOR / XNOR conventions), and no single item can be read off
  it alone.
* **general** -- ``f`` depends on two or more items but is not affine
  (e.g. AND, OR, majority): a nonlinear holistic rule.

The *support* of ``f`` (its set of relevant variables) says exactly which
guessers' items a given light actually carries information about, which is
the structural counterpart to the mutual-information numbers in
:mod:`gsg.evaluation.information_theory`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Observation = Tuple[int, ...]


@dataclass
class SignallerFunction:
    """Structural classification of one signaller's Boolean policy."""

    n_inputs: int
    support: Tuple[int, ...]   # guesser labels the bit actually depends on
    kind: str                  # "constant" | "dictator" | "parity" | "general"
    name: str                  # human-readable description


def _relevant_variables(policy: Dict[Observation, int], n: int) -> Tuple[int, ...]:
    """Positions ``k`` for which flipping input ``k`` ever changes the output."""
    relevant = []
    for k in range(n):
        for obs, out in policy.items():
            flipped = list(obs)
            flipped[k] = 1 - flipped[k]
            if policy[tuple(flipped)] != out:
                relevant.append(k)
                break
    return tuple(relevant)


def _affine_coeffs(policy: Dict[Observation, int], n: int):
    """If ``f`` is affine over GF(2) return ``(c, coeffs)`` with
    ``f(x) = c (+) sum_k coeffs[k]*x_k``; otherwise return ``None``."""
    zero = tuple(0 for _ in range(n))
    c = policy[zero]
    coeffs = []
    for k in range(n):
        e_k = tuple(1 if i == k else 0 for i in range(n))
        coeffs.append(policy[e_k] ^ c)
    # verify against every input
    for obs, out in policy.items():
        pred = c
        for k in range(n):
            if coeffs[k]:
                pred ^= obs[k]
        if pred != out:
            return None
    return c, coeffs


def analyse_signaller_policy(
    policy: Dict[Observation, int], neighbour_labels: Sequence[int]
) -> SignallerFunction:
    """Classify one signaller's greedy Boolean policy.

    ``policy`` maps each observation (a tuple of the neighbouring guessers'
    items, in ``neighbour_labels`` order) to the emitted bit -- exactly the
    dict :func:`gsg.evaluation.conventions.greedy_policy` returns.
    ``neighbour_labels`` gives the guesser id at each observation position
    (``graph.out_neighbours(signaller)``), so the returned support and name
    are stated in terms of guesser ids, not raw positions.
    """
    n = len(neighbour_labels)
    support_pos = _relevant_variables(policy, n)
    support = tuple(neighbour_labels[k] for k in support_pos)

    # constant / pooling
    if not support_pos:
        const = next(iter(policy.values()))
        return SignallerFunction(
            n_inputs=n, support=(), kind="constant",
            name=f"constant {'ON' if const else 'OFF'} (pooling)",
        )

    affine = _affine_coeffs(policy, n)

    # dictator: depends on exactly one item
    if len(support_pos) == 1:
        k = support_pos[0]
        g = neighbour_labels[k]
        # f restricted to x_k: is it x_k (track) or 1-x_k (track NOT)?
        e_k = tuple(1 if i == k else 0 for i in range(n))
        zero = tuple(0 for _ in range(n))
        emits_on_for_dog = policy[e_k] == 1 and policy[zero] == 0
        name = f"track G{g}" if emits_on_for_dog else f"track NOT G{g}"
        return SignallerFunction(
            n_inputs=n, support=support, kind="dictator", name=name,
        )

    # parity: affine in two or more variables
    if affine is not None:
        c, _ = affine
        inside = ",".join(f"G{g}" for g in support)
        base = f"XOR({inside})"
        name = base if c == 0 else f"XNOR({inside}) [= NOT {base}]"
        return SignallerFunction(
            n_inputs=n, support=support, kind="parity", name=name,
        )

    # general nonlinear
    inside = ",".join(f"G{g}" for g in support)
    return SignallerFunction(
        n_inputs=n, support=support, kind="general",
        name=f"general({inside}) [nonlinear]",
    )


def theoretical_convention_counts(m: int) -> Dict[str, int]:
    """Sizes of the K_{m,m} signaller-convention space, for context.

    * ``policies_per_signaller`` = ``2 ** (2 ** m)`` -- Boolean functions on m
      binary inputs.
    * ``joint_conventions`` = ``policies_per_signaller ** m``.
    * ``perfect_conventions`` = ``(2 ** m)!`` -- bijective joint encodings of
      the ``2 ** m`` worlds (the R = 1.0 set).
    * ``tracking_conventions`` = ``m! * 2 ** m`` -- the compositional subset:
      a bijection assigning each signaller one guesser, times a polarity per
      signaller.
    """
    import math

    w = 2 ** m
    return {
        "policies_per_signaller": 2 ** w,
        "joint_conventions": (2 ** w) ** m,
        "perfect_conventions": math.factorial(w),
        "tracking_conventions": math.factorial(m) * (2 ** m),
    }


def classify_convention(
    functions: Dict[int, SignallerFunction], num_guessers: int, is_perfect: bool
) -> str:
    """One-line summary of a whole population's convention structure.

    ``functions`` maps each signaller id to its
    :func:`analyse_signaller_policy` result.
    """
    kinds = [f.kind for f in functions.values()]
    all_dictators = all(k == "dictator" for k in kinds)
    tracked = {f.support[0] for f in functions.values() if f.kind == "dictator"}
    covers_all = len(tracked) == num_guessers

    if is_perfect and all_dictators and covers_all:
        return "PERFECT pure-tracking (fully compositional: every light = one guesser's item)"
    if is_perfect:
        holistic = [f"S{s}={f.name}" for s, f in functions.items() if f.kind in ("parity", "general")]
        return f"PERFECT holistic (uses relational encoding: {'; '.join(holistic)})"
    n_pool = kinds.count("constant")
    n_holistic = kinds.count("parity") + kinds.count("general")
    return (f"not perfect ({kinds.count('dictator')} tracking, {n_holistic} holistic, "
            f"{n_pool} pooling signallers)")
