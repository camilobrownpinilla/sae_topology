"""Tests that `precomputed_knn` produces identical eigenvalues / eigenvectors
to a fresh kNN computation (within the eigenvector sign ambiguity)."""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.spectral import (
    build_coifman_lafon,
    coifman_lafon_spectrum,
    compute_knn,
)


def _aligned(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Flip column signs of v_b so each column matches v_a's direction."""
    out = v_b.copy()
    for j in range(v_b.shape[1]):
        if np.dot(v_a[:, j], v_b[:, j]) < 0:
            out[:, j] = -out[:, j]
    return out


def test_compute_knn_shapes(circle_small):
    X, _ = circle_small
    d, i = compute_knn(X, knn_k=15)
    assert d.shape == (len(X), 15)
    assert i.shape == (len(X), 15)
    # self should not appear in the kNN (we strip column 0 internally)
    for row in range(len(X)):
        assert row not in i[row]


def test_precomputed_knn_matches_fresh_eigenvalues(circle_small):
    X, _ = circle_small
    spec_fresh = coifman_lafon_spectrum(X, knn_k=15, K=8)
    d, i = compute_knn(X, knn_k=15)
    spec_cached = coifman_lafon_spectrum(
        X, knn_k=15, K=8, precomputed_knn=(d, i),
    )
    np.testing.assert_allclose(
        spec_fresh['eigenvalues'], spec_cached['eigenvalues'],
        rtol=1e-12, atol=1e-12,
    )


def test_precomputed_knn_matches_fresh_eigenvectors(circle_small):
    X, _ = circle_small
    spec_fresh = coifman_lafon_spectrum(X, knn_k=15, K=8)
    d, i = compute_knn(X, knn_k=15)
    spec_cached = coifman_lafon_spectrum(
        X, knn_k=15, K=8, precomputed_knn=(d, i),
    )
    aligned = _aligned(spec_fresh['eigenvectors'], spec_cached['eigenvectors'])
    np.testing.assert_allclose(
        spec_fresh['eigenvectors'], aligned, rtol=1e-10, atol=1e-10,
    )


def test_precomputed_knn_invariance_across_sigma_factor(circle_small):
    """The same (Z, knn_k) kNN can be reused across sigma_factor variants —
    the graph topology depends only on (Z, knn_k)."""
    X, _ = circle_small
    d, i = compute_knn(X, knn_k=15)
    for sf in [0.5, 1.0, 2.0]:
        spec_fresh = coifman_lafon_spectrum(X, knn_k=15, K=6, sigma_factor=sf)
        spec_cached = coifman_lafon_spectrum(
            X, knn_k=15, K=6, sigma_factor=sf, precomputed_knn=(d, i),
        )
        np.testing.assert_allclose(
            spec_fresh['eigenvalues'], spec_cached['eigenvalues'],
            rtol=1e-12, atol=1e-12,
            err_msg=f'sigma_factor={sf}',
        )


def test_precomputed_knn_shape_validation(circle_small):
    X, _ = circle_small
    d, i = compute_knn(X, knn_k=15)
    # Wrong knn_k raises
    with pytest.raises(ValueError, match='shape mismatch'):
        build_coifman_lafon(X, knn_k=10, precomputed_knn=(d, i))
    # Wrong N raises
    with pytest.raises(ValueError, match='shape mismatch'):
        build_coifman_lafon(X, knn_k=15, precomputed_knn=(d[:100], i[:100]))
