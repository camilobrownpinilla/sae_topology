"""Tests that the parallel and serial paths produce identical results.

We exercise the smallest possible inputs that still cover the parallel
dispatch logic. Tests are gated to `n_jobs=2` to keep memory pressure low
on the development machine; the design supports higher n_jobs but the
correctness invariants are the same.
"""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.mapper import mapper_sweep
from sae_topology.mapper.filters import laplacian_eigenvector_filter
from sae_topology.spectral import coifman_lafon_spectrum


# Tiny grids — keep parallelization tests fast and memory-light.
TINY_NI_GRID = (5, 8)
TINY_OV_GRID = (0.25, 0.50)


def _entries_equal(a: dict, b: dict) -> None:
    """Assert two mapper_sweep result dicts are equal at every (ni, ov)."""
    assert set(a.keys()) == set(b.keys()), (
        f'key sets differ: {set(a.keys()) ^ set(b.keys())}'
    )
    for key in a:
        ea, eb = a[key], b[key]
        assert ea.keys() == eb.keys(), f'{key}: subkey sets differ'
        for k in ea:
            va, vb = ea[k], eb[k]
            if isinstance(va, dict):
                assert va == vb, f'{key}/{k}: dict mismatch {va} vs {vb}'
            elif isinstance(va, float):
                assert (va == vb) or (np.isnan(va) and np.isnan(vb)), (
                    f'{key}/{k}: float mismatch {va} vs {vb}'
                )
            else:
                assert va == vb, f'{key}/{k}: mismatch {va} vs {vb}'


def _filter_for(X: np.ndarray, k_filter: int):
    """Compute Laplacian eigenvector filter values for a small point cloud."""
    spec = coifman_lafon_spectrum(X, knn_k=15, K=k_filter + 1)
    return laplacian_eigenvector_filter(
        X, k=k_filter, eigenvectors=spec['eigenvectors'],
    )


def test_mapper_sweep_serial_equals_parallel_circle(circle_small):
    X, _ = circle_small
    lens = _filter_for(X, k_filter=2)
    serial = mapper_sweep(
        X, lens, topology='circle',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=1,
    )
    parallel = mapper_sweep(
        X, lens, topology='circle',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=2,
    )
    _entries_equal(serial, parallel)


def test_mapper_sweep_serial_equals_parallel_torus(torus_small):
    X, _ = torus_small
    lens = _filter_for(X, k_filter=3)
    serial = mapper_sweep(
        X, lens, topology='torus',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=1,
    )
    parallel = mapper_sweep(
        X, lens, topology='torus',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=2,
    )
    _entries_equal(serial, parallel)


def test_mapper_sweep_no_topology_argument(circle_small):
    """When topology is None, entries omit nerve_mismatch — must match in
    both serial and parallel paths."""
    X, _ = circle_small
    lens = _filter_for(X, k_filter=2)
    serial = mapper_sweep(
        X, lens, topology=None,
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=1,
    )
    parallel = mapper_sweep(
        X, lens, topology=None,
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        n_jobs=2,
    )
    _entries_equal(serial, parallel)
    # Verify nerve_mismatch is genuinely absent
    for key, entry in serial.items():
        assert 'nerve_mismatch' not in entry


def test_mapper_sweep_distance_threshold_propagates(circle_small):
    """Passing an explicit distance_threshold should give the same results
    in serial and parallel paths."""
    X, _ = circle_small
    lens = _filter_for(X, k_filter=2)
    serial = mapper_sweep(
        X, lens, topology='circle',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        distance_threshold=0.5, n_jobs=1,
    )
    parallel = mapper_sweep(
        X, lens, topology='circle',
        n_intervals_grid=TINY_NI_GRID, overlap_grid=TINY_OV_GRID,
        distance_threshold=0.5, n_jobs=2,
    )
    _entries_equal(serial, parallel)
