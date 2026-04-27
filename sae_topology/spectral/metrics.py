"""Comparison metrics between empirical (graph-Laplacian) and theoretical
(closed-form Laplace-Beltrami) spectra. Per spec section 4.4.

Empirical eigenvalues have arbitrary scale (depends on N, sigma, embedding),
so all comparisons are scale-invariant: log-ratio error and additive-window
multiplicity matching after normalisation by lambda_1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def log_ratio_error(emp: np.ndarray, theory: np.ndarray) -> float:
    """Primary metric (spec section 4.4):

        E = (1 / (K - 1)) sum_{i=2}^{K} | log(emp_i / emp_1) - log(theory_i / theory_1) |

    where emp_1, theory_1 are the first non-zero eigenvalue of each spectrum
    (the leading zero eigenvalue is skipped). Both arrays must be sorted
    ascending and must have at least 3 entries.

    Returns inf if theory or emp contains a non-positive value past the
    leading zero (signals collapse / disconnection).
    """
    emp = np.sort(np.asarray(emp, dtype=float))
    theory = np.sort(np.asarray(theory, dtype=float))
    K = min(len(emp), len(theory))
    if K < 3:
        raise ValueError("log_ratio_error needs at least K=3 eigenvalues.")

    # Skip the leading zero eigenvalue from both spectra (index 0).
    emp_use = emp[1:K]
    theory_use = theory[1:K]
    if (emp_use[0] <= 0.0) or (theory_use[0] <= 0.0):
        return float('inf')

    emp_log = np.log(emp_use[1:] / emp_use[0])
    theory_log = np.log(theory_use[1:] / theory_use[0])
    return float(np.mean(np.abs(emp_log - theory_log)))


def multiplicity_check(emp: np.ndarray, theory_levels: np.ndarray,
                       eps: float = 0.10) -> dict:
    """Count empirical eigenvalues within a tolerance window around each
    theoretical level (after normalising emp by emp_1, the first non-zero eig).

    Window is `max(eps, eps * |L|)` so the same eps gives a fixed additive
    width near zero and a multiplicative width away from zero. Default
    eps=0.10 is a 10% relative window, tunable during Stage 0 (spec section 9).

    Returns dict {level -> empirical_count} suitable for direct comparison
    against the theoretical multiplicity vector.
    """
    emp = np.sort(np.asarray(emp, dtype=float))
    if len(emp) < 2:
        raise ValueError("multiplicity_check needs at least 2 eigenvalues.")

    nonzero = emp[emp > 1e-10]
    if len(nonzero) == 0:
        raise ValueError("All empirical eigenvalues are ~0; cannot normalise.")
    lambda1 = nonzero[0]
    emp_norm = emp / lambda1

    counts: dict = {}
    for L in np.asarray(theory_levels, dtype=float):
        window = max(eps, eps * abs(L))
        counts[float(L)] = int((np.abs(emp_norm - L) < window).sum())
    return counts


def near_zero_count(emp: np.ndarray, threshold: float = 1e-4) -> int:
    """Count empirical eigenvalues below `threshold`. Approximates b_0 (number
    of connected components) of the underlying manifold.

    Default threshold 1e-4 cleanly distinguishes numerical-zero eigenvalues
    (machine precision ~1e-15) from a typical lambda_1 (>=~1e-3 for our
    Coifman-Lafon graph Laplacian on Stage-0 sample sizes).
    """
    return int((np.asarray(emp, dtype=float) < threshold).sum())


def first_nonzero_eigenvalue(emp: np.ndarray, zero_threshold: float = 1e-4) -> Optional[float]:
    """Return the smallest empirical eigenvalue exceeding `zero_threshold`,
    or None if all are below threshold."""
    nonzero = np.asarray(emp, dtype=float)
    nonzero = nonzero[nonzero > zero_threshold]
    if len(nonzero) == 0:
        return None
    return float(np.sort(nonzero)[0])
