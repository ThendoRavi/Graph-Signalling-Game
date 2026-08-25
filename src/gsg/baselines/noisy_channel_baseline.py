"""Noisy channel baseline (Section 4.5.3).

Frozen trained policies evaluated under a binary symmetric channel
with flip probability p in {0.0, 0.1, 0.2, 0.3}. Tests convention
robustness (claim ii); robust if reward at p=0.2 stays >= 0.75.
"""
