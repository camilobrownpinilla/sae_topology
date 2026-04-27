"""Tests for the sparse eigsh robustness cascade in
`laplacian_eigendecomposition`.

The cascade:
  1. eigsh(SA, v0=ones/sqrt(N))             -- primary path
  2. eigsh(shift-invert sigma=0)             -- on ArpackNoConvergence
  3. eigsh(SA, maxiter=5000, tol=1e-9)       -- on second failure
  4. dense eigh + warn (only if N < DENSE_FALLBACK_MAX_N)
  4'. RuntimeError                            (if N >= DENSE_FALLBACK_MAX_N)
"""
from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sae_topology.spectral import (
    DENSE_FALLBACK_MAX_N,
    coifman_lafon_spectrum,
    laplacian_eigendecomposition,
)


def _disconnected_blocks(N: int, n_blocks: int) -> sp.csr_matrix:
    """Build a normalised Laplacian with `n_blocks` disconnected components.
    The Laplacian has eigenvalue 0 with multiplicity `n_blocks` — a stress
    test for ARPACK near the smallest-algebraic end."""
    block_size = N // n_blocks
    blocks = []
    for _ in range(n_blocks):
        # path graph on block_size nodes
        diag = np.full(block_size, 1.0)
        diag[0] = 0.5
        diag[-1] = 0.5
        off = -0.5 * np.ones(block_size - 1)
        b = sp.diags([diag, off, off], [0, 1, -1], format='csr')
        blocks.append(b)
    return sp.block_diag(blocks).tocsr()


def test_primary_path_succeeds_on_normal_laplacian():
    """Path 1 should handle a generic Laplacian without falling through."""
    N = 300
    L = _disconnected_blocks(N, n_blocks=3)
    out = laplacian_eigendecomposition(L, K=6)
    eigs = out['eigenvalues']
    # 3 blocks → 3 zero eigenvalues
    np.testing.assert_array_less(eigs[:3], 1e-6)
    assert eigs[3] > 1e-4
    assert out['eigenvectors'].shape == (N, 6)


def test_eigenvalues_are_sorted_and_nonnegative():
    N = 200
    L = _disconnected_blocks(N, n_blocks=2)
    out = laplacian_eigendecomposition(L, K=8)
    eigs = out['eigenvalues']
    assert (eigs[:-1] <= eigs[1:] + 1e-12).all()
    assert (eigs >= 0).all()


def test_shift_invert_path_used_when_primary_fails():
    """Mock primary eigsh to raise; verify the shift-invert call happens."""
    N = 200
    L = _disconnected_blocks(N, n_blocks=2)
    real_eigsh = spla.eigsh
    call_log = []

    def fake_eigsh(*args, **kwargs):
        call_log.append(kwargs)
        if len(call_log) == 1:
            raise spla.ArpackNoConvergence('mock', np.array([]), np.array([]))
        return real_eigsh(*args, **kwargs)

    with mock.patch('sae_topology.spectral.graph_laplacian.spla.eigsh',
                    side_effect=fake_eigsh):
        out = laplacian_eigendecomposition(L, K=4)

    # First call: primary SA path. Second: shift-invert.
    assert len(call_log) >= 2
    assert call_log[1].get('sigma') == 0.0
    assert call_log[1].get('mode') == 'normal'
    assert out['eigenvalues'].shape == (4,)


def test_dense_fallback_warns_below_threshold():
    """Below DENSE_FALLBACK_MAX_N: dense fallback fires with a warning."""
    N = 150
    L = _disconnected_blocks(N, n_blocks=2)

    def always_fail(*args, **kwargs):
        raise spla.ArpackNoConvergence('mock', np.array([]), np.array([]))

    with mock.patch('sae_topology.spectral.graph_laplacian.spla.eigsh',
                    side_effect=always_fail):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = laplacian_eigendecomposition(L, K=4)
            assert any('falling back to dense' in str(w.message) for w in caught)
    assert out['eigenvalues'].shape == (4,)


def test_dense_fallback_blocked_above_threshold():
    """At N >= DENSE_FALLBACK_MAX_N: refuse dense, raise RuntimeError.

    We don't actually allocate a 20k Laplacian (too slow for unit tests);
    instead we mock both eigsh and the .shape attribute lookup."""
    N = DENSE_FALLBACK_MAX_N  # exactly at the threshold -> blocked

    fake_L = mock.MagicMock(spec=sp.csr_matrix)
    fake_L.shape = (N, N)

    def always_fail(*args, **kwargs):
        raise spla.ArpackNoConvergence('mock', np.array([]), np.array([]))

    with mock.patch('sae_topology.spectral.graph_laplacian.spla.eigsh',
                    side_effect=always_fail):
        with pytest.raises(RuntimeError, match='dense fallback is disabled'):
            laplacian_eigendecomposition(fake_L, K=4)
    # Confirm .toarray was never called (no dense allocation attempted).
    fake_L.toarray.assert_not_called()


def test_v0_is_deterministic_across_calls():
    """Calling eigsh twice on the same L should give numerically identical
    eigenvalues because v0 is constant (not RNG-derived). Tolerance allows
    for BLAS reduction-order noise (~machine epsilon).

    Eigenvectors are not compared: ARPACK is free to return any
    orthonormal basis within a degenerate eigenspace, so equality there is
    not a meaningful test."""
    N = 300
    L = _disconnected_blocks(N, n_blocks=3)
    out_a = laplacian_eigendecomposition(L, K=6)
    out_b = laplacian_eigendecomposition(L, K=6)
    np.testing.assert_allclose(
        out_a['eigenvalues'], out_b['eigenvalues'], rtol=1e-12, atol=1e-14,
    )


def test_end_to_end_spectrum_on_real_data():
    """Sanity: full pipeline on a real-ish point cloud completes and returns
    finite eigenvalues."""
    np.random.seed(0)
    # 2 well-separated Gaussian blobs in R^4
    X = np.vstack([
        np.random.randn(150, 4),
        np.random.randn(150, 4) + 5.0,
    ])
    spec = coifman_lafon_spectrum(X, knn_k=10, K=6)
    assert np.all(np.isfinite(spec['eigenvalues']))
    assert spec['eigenvalues'].shape == (6,)
