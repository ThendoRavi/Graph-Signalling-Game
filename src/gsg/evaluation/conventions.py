"""Convention extraction and enumeration (Section 4.6.3).

Extracts each signaller's greedy policy pi_s: {0,1}^2 -> {0,1},
maps it to a canonical convention id (16 per signaller), and forms
the joint convention pair (pi_s1, pi_s2). Supports the ternary
variant (512 mappings, Section 4.6.6).
"""
