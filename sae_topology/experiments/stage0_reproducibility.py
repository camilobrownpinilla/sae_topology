"""Spectral-only cross-seed reproducibility check for $T^2$ and $S^2$.

Per stage0_tuning.md §10 sign-off checklist:
    "Eigenvector-basis-rotation reproducibility check passes for T^2 and S^2
     (Mapper Betti unchanged across random seeds despite eigenvector basis
     varying within degenerate eigenspaces)."

Per user direction this turn: scrap the full noise-perturbation robustness
check (§5). Keep ONLY a *spectral* version of the cross-seed reproducibility
check on T^2 and S^2: re-run the Coifman-Lafon spectrum at a second seed,
verify the eigenvalue ratios (and consequently log-ratio error and
multiplicity-cluster check) are stable. The Mapper-graph half of the spec
check is dropped here.

T^2 (mult of lambda_1 is 4) and S^2 (mult is 3) have nontrivial rotation
freedom in their first-eigenspace basis. The eigenvalues themselves should
not depend on this gauge — but small numerical perturbations in the kNN
graph can move them. This module confirms they don't.

S^1 is excluded: its lambda_1 has multiplicity 2, which is a 2D eigenspace,
but the determined-up-to-sign cos/sin pair is canonical enough that
reproducibility is generally not in question for this manifold.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from sae_topology.dgp import make_dgp
from sae_topology.spectral import (
    coifman_lafon_spectrum,
    log_ratio_error,
    multiplicity_clusters_match,
    REFERENCE_SPECTRA,
)


REPRODUCIBILITY_TOPOLOGIES = ('torus', 'sphere')


def _dgp_kwargs(topology: str) -> dict:
    if topology == 'torus':
        return {'major_radius': 1.0, 'minor_radius': 1.0}
    if topology == 'sphere':
        return {'radius': 1.0}
    return {}


def _sample(topology: str, n: int, ambient_d: int, sigma: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    dgp = make_dgp(topology, d=ambient_d, sigma=sigma,
                   c0=np.zeros(ambient_d), **_dgp_kwargs(topology))
    return dgp.sample_with_gt(n)[0]


def run_spectral_reproducibility(
    topology: str,
    n_samples: int,
    ambient_d: int,
    sigma_data: float,
    knn_k: int,
    sigma_factor: float,
    K: int = 20,
    seeds: Sequence[int] = (0, 1),
    log_ratio_tolerance: float = 0.02,
    multiplicity_eps: float = 0.05,
    multiplicity_n_clusters: int = 4,
) -> dict:
    """Re-run the Coifman-Lafon spectrum at each seed; verify spectral E and
    multiplicity-cluster check are stable across seeds.

    Args:
        topology:               must be in REPRODUCIBILITY_TOPOLOGIES.
        n_samples, ambient_d,
        sigma_data:             same DGP knobs as Stage 0 used for the winner.
        knn_k, sigma_factor:    the winning auto-tune anchor.
        K:                      number of eigenvalues to compute.
        seeds:                  the two seeds to compare. The first is
                                conventionally the same seed Stage 0 used
                                (so the spectrum from Stage 0 is reproduced
                                here).
        log_ratio_tolerance:    pass if |E[seed_a] - E[seed_b]| <= this.

    Returns dict:
        per_seed: {seed -> {eigenvalues, log_ratio_error, multiplicity_check}}
        log_ratio_error_diff:  max | E[seed_i] - E[seed_j] | across seed pairs
        all_multiplicity_pass: every per-seed multiplicity_clusters_match
                               returns all_match=True
        pass:                  log_ratio_error_diff <= log_ratio_tolerance
                               AND all_multiplicity_pass
    """
    if topology not in REPRODUCIBILITY_TOPOLOGIES:
        raise ValueError(
            f"Spectral reproducibility check is only meaningful for "
            f"degenerate-lambda_1 manifolds: {REPRODUCIBILITY_TOPOLOGIES}. "
            f"Got topology={topology!r}."
        )
    if topology not in REFERENCE_SPECTRA:
        raise ValueError(f"No closed-form spectrum for topology {topology!r}.")

    ref = REFERENCE_SPECTRA[topology]
    per_seed: dict[int, dict] = {}

    for seed in seeds:
        X = _sample(topology, n_samples, ambient_d, sigma_data, int(seed))
        spec = coifman_lafon_spectrum(
            X, knn_k=knn_k, K=K, sigma_factor=sigma_factor,
        )
        eigs = np.asarray(spec['eigenvalues'], dtype=float)
        try:
            E = float(log_ratio_error(eigs, ref['ratios']))
        except Exception:
            E = float('inf')
        try:
            mc = multiplicity_clusters_match(
                eigs, ref['levels'], ref['mults'],
                eps=multiplicity_eps, n_clusters=multiplicity_n_clusters,
            )
        except Exception as e:
            mc = {'error': str(e), 'all_match': False, 'per_cluster': []}
        per_seed[int(seed)] = {
            'eigenvalues': [float(v) for v in eigs],
            'log_ratio_error': E,
            'multiplicity_check': mc,
        }

    Es = [v['log_ratio_error'] for v in per_seed.values()
          if v['log_ratio_error'] is not None and np.isfinite(v['log_ratio_error'])]
    if len(Es) >= 2:
        E_diff = max(Es) - min(Es)
    else:
        E_diff = float('inf')

    all_mult_pass = all(
        v['multiplicity_check'].get('all_match', False) for v in per_seed.values()
    )
    passed = bool(E_diff <= log_ratio_tolerance and all_mult_pass)

    return {
        'topology': topology,
        'seeds': list(map(int, seeds)),
        'per_seed': per_seed,
        'log_ratio_error_diff': E_diff,
        'log_ratio_tolerance': log_ratio_tolerance,
        'all_multiplicity_pass': bool(all_mult_pass),
        'pass': passed,
    }
