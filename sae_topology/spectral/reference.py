"""Closed-form Laplace-Beltrami spectra and ground-truth Betti numbers.

Per spec section 4.5. All eigenvalues are normalized by lambda_1 (the
first non-zero eigenvalue) so they can be compared to graph-Laplacian
eigenvalues which carry an arbitrary scale (depends on N, sigma,
embedding).

Conventions:
  *_LEVELS: unique eigenvalue ratios, sorted ascending.
  *_MULTS:  multiplicity of each level in *_LEVELS.
  *_RATIOS: ratios expanded to a flat list (length = sum(MULTS)).

Use *_RATIOS to compare against the empirical eigenvalue list
(also expanded). Use *_LEVELS / *_MULTS for the multiplicity check.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Ground-truth Betti numbers (relocated from sae_topology/ph/reference.py)
# ----------------------------------------------------------------------

GROUND_TRUTH_BETTI: dict = {
    'points':       {'H0': 3, 'H1': 0},
    'circle':       {'H0': 1, 'H1': 1},
    'two_circles':  {'H0': 2, 'H1': 2},
    'figure_eight': {'H0': 1, 'H1': 2},
    'torus':        {'H0': 1, 'H1': 2, 'H2': 1},
    'sphere':       {'H0': 1, 'H1': 0, 'H2': 1},
    'helix':        {'H0': 1, 'H1': 0},
}


# ----------------------------------------------------------------------
# S^1 (unit circle): lambda_k = k^2, multiplicity 2 for k > 0
# ----------------------------------------------------------------------

def _s1_spectrum(k_max: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Levels and multiplicities for S^1 up to wavenumber k_max."""
    levels = np.array([k * k for k in range(k_max + 1)], dtype=float)
    mults = np.array([1] + [2] * k_max, dtype=int)
    return levels, mults


S1_LEVELS, S1_MULTS = _s1_spectrum(k_max=20)
S1_RATIOS = np.repeat(S1_LEVELS, S1_MULTS)  # divide by lambda_1 = 1 -> identity


# ----------------------------------------------------------------------
# T^2 (unit flat torus, [0, 2pi)^2): lambda_{m,n} = m^2 + n^2,
# multiplicity = r_2(lambda) = #{(m, n) in Z^2 : m^2 + n^2 = lambda}
# ----------------------------------------------------------------------

def _t2_spectrum(M: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Levels and multiplicities for T^2, scanning |m|, |n| <= M."""
    counts: dict[int, int] = {}
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            level = m * m + n * n
            counts[level] = counts.get(level, 0) + 1
    levels_sorted = sorted(counts.keys())
    levels = np.array(levels_sorted, dtype=float)
    mults = np.array([counts[int(L)] for L in levels_sorted], dtype=int)
    return levels, mults


T2_LEVELS, T2_MULTS = _t2_spectrum(M=8)
T2_RATIOS = np.repeat(T2_LEVELS, T2_MULTS)
# T^2 ratios already normalised by lambda_1 = 1.


# ----------------------------------------------------------------------
# S^2 (unit sphere): lambda_ell = ell(ell+1), multiplicity 2*ell + 1.
# Reported in ratios divided by lambda_1 = 2 -> [0, 1, 3, 6, 10, 15, 21, ...]
# ----------------------------------------------------------------------

def _s2_spectrum(ell_max: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Levels (in lambda/lambda_1 ratio) and multiplicities for S^2."""
    raw = np.array([ell * (ell + 1) for ell in range(ell_max + 1)], dtype=float)
    lambda1 = 2.0
    levels = raw / lambda1                              # 0, 1, 3, 6, 10, ...
    mults = np.array([2 * ell + 1 for ell in range(ell_max + 1)], dtype=int)
    return levels, mults


S2_LEVELS, S2_MULTS = _s2_spectrum(ell_max=12)
S2_RATIOS = np.repeat(S2_LEVELS, S2_MULTS)


# ----------------------------------------------------------------------
# Line segment with Neumann BC: lambda_n = (n*pi/L)^2, all multiplicity 1.
# Normalized by lambda_1: levels = [0, 1, 4, 9, ...] = [n^2 for n in 0..].
# Same numerical levels as S^1 but multiplicities all 1, not 2.
# Used as the reference for the Helix DGP: the helix is Riemannian-isometric
# to a line segment of arc length L = 2*pi*n_turns*sqrt(R^2 + c^2).
# ----------------------------------------------------------------------

def _line_spectrum(n_max: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Levels and multiplicities for a line segment with Neumann BC."""
    levels = np.array([n * n for n in range(n_max + 1)], dtype=float)
    mults = np.ones(n_max + 1, dtype=int)
    return levels, mults


LINE_LEVELS, LINE_MULTS = _line_spectrum(n_max=20)
LINE_RATIOS = np.repeat(LINE_LEVELS, LINE_MULTS)  # = LINE_LEVELS since all mult 1.

# Helix is intrinsically a line; expose aliases for self-documenting downstream use.
HELIX_LEVELS = LINE_LEVELS
HELIX_MULTS = LINE_MULTS
HELIX_RATIOS = LINE_RATIOS


# ----------------------------------------------------------------------
# Lookup helpers
# ----------------------------------------------------------------------

REFERENCE_SPECTRA: dict = {
    'circle': {'levels': S1_LEVELS, 'mults': S1_MULTS, 'ratios': S1_RATIOS,
               'gap_levels': []},
    'torus':  {'levels': T2_LEVELS, 'mults': T2_MULTS, 'ratios': T2_RATIOS,
               'gap_levels': [3, 6, 7, 11, 12, 14, 15]},
    'sphere': {'levels': S2_LEVELS, 'mults': S2_MULTS, 'ratios': S2_RATIOS,
               'gap_levels': []},
    'helix':  {'levels': LINE_LEVELS, 'mults': LINE_MULTS, 'ratios': LINE_RATIOS,
               'gap_levels': []},
}


def reference_ratios(topology: str, K: int = 20) -> np.ndarray:
    """Return the first K theoretical ratios for the topology, including the zero
    eigenvalue. Caller is responsible for slicing K appropriately for comparison."""
    spec = REFERENCE_SPECTRA.get(topology)
    if spec is None:
        raise ValueError(
            f"No closed-form spectrum for topology '{topology}'. "
            f"Available: {list(REFERENCE_SPECTRA.keys())}"
        )
    ratios = spec['ratios']
    return ratios[:K] if K is not None else ratios
