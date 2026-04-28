"""Smoke tests for run_spectral_reproducibility (T^2 / S^2)."""
from __future__ import annotations

import pytest

from sae_topology.experiments.stage0_reproducibility import (
    REPRODUCIBILITY_TOPOLOGIES,
    run_spectral_reproducibility,
)


@pytest.mark.parametrize('topology', list(REPRODUCIBILITY_TOPOLOGIES))
def test_spectral_reproducibility_smoke(topology):
    """Tiny N: just verify the function runs end-to-end and returns the
    expected dict shape. Don't enforce pass/fail at small N — at N=600
    the multiplicity check often fails for T²/S² regardless of seed."""
    out = run_spectral_reproducibility(
        topology=topology,
        n_samples=600,
        ambient_d=8,
        sigma_data=0.05,
        knn_k=15,
        sigma_factor=1.0,
        K=10,
        seeds=(0, 1),
    )
    assert out['topology'] == topology
    assert set(out['per_seed'].keys()) == {0, 1}
    for seed, payload in out['per_seed'].items():
        assert 'eigenvalues' in payload
        assert isinstance(payload['log_ratio_error'], float)
        assert 'multiplicity_check' in payload
    assert isinstance(out['log_ratio_error_diff'], float)
    assert isinstance(out['pass'], bool)


def test_reproducibility_rejects_non_degenerate_topology():
    """S^1 has lambda_1 mult=2 BUT the cos/sin pair is canonical up to sign,
    so we don't run the gauge-rotation check for it. The function must
    raise on circle/figure_eight."""
    with pytest.raises(ValueError):
        run_spectral_reproducibility(
            topology='circle', n_samples=200, ambient_d=8,
            sigma_data=0.05, knn_k=15, sigma_factor=1.0, K=8,
        )
    with pytest.raises(ValueError):
        run_spectral_reproducibility(
            topology='figure_eight', n_samples=200, ambient_d=8,
            sigma_data=0.05, knn_k=15, sigma_factor=1.0, K=8,
        )
