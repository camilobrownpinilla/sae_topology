"""Determinism test for the flat Stage B pool added in this revision.

`run_stage0` now dispatches Stage B as a single 216-job loky pool (one
job per (topology, filter, n_intervals, overlap)), which is a refactor
of the prior 11-job pool (one job per (topology, filter), each running
the full grid sequentially). The new path must remain bit-deterministic
between serial and parallel dispatch.

This is partially redundant with `test_stage0_determinism`; the existing
end-to-end test already passes, but here we lock down the per-cell
sweep entries explicitly.
"""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.experiments.stage0_validate import run_stage0


SMALL_KWARGS = dict(
    ambient_d=8,
    sigma=0.05,
    seed=0,
    knn_k_grid=(15, 25),
    sigma_factor_grid=(0.5, 1.0),
    n_intervals_grid=(5, 8, 12),
    overlap_grid=(0.25, 0.35),
    inner_n_intervals_grid=(5, 8),
    spectral_K=8,
    n_samples_override={'circle': 600},
)


def _entries_equal(a: dict, b: dict) -> None:
    """Compare two per-(ni, ov) sweep dicts entry-wise."""
    assert set(a.keys()) == set(b.keys())
    for key in a:
        ea, eb = a[key], b[key]
        for k in ea:
            va, vb = ea[k], eb[k]
            if isinstance(va, dict):
                assert va == vb, f'{key}/{k}: {va!r} vs {vb!r}'
            elif isinstance(va, float):
                if np.isnan(va):
                    assert np.isnan(vb)
                else:
                    assert va == vb, f'{key}/{k}: {va} != {vb}'
            else:
                assert va == vb, f'{key}/{k}: {va!r} vs {vb!r}'


def test_flat_stage_b_serial_equals_parallel():
    """The 18-job flat pool (1 topology × 3 filters × 3 ni × 2 ov) must yield
    identical per-cell entries between serial and 4-way parallel dispatch."""
    sweeps_serial: dict[str, dict[str, dict]] = {}
    sweeps_parallel: dict[str, dict[str, dict]] = {}

    run_stage0(
        topologies=('circle',), n_jobs=1,
        capture_sweeps_into=sweeps_serial,
        **SMALL_KWARGS,
    )
    run_stage0(
        topologies=('circle',), n_jobs=4,
        capture_sweeps_into=sweeps_parallel,
        **SMALL_KWARGS,
    )

    assert set(sweeps_serial.keys()) == set(sweeps_parallel.keys())
    for topo in sweeps_serial:
        assert set(sweeps_serial[topo].keys()) == set(sweeps_parallel[topo].keys())
        for fk in sweeps_serial[topo]:
            _entries_equal(sweeps_serial[topo][fk], sweeps_parallel[topo][fk])


def test_flat_stage_b_winners_match():
    """Winners (knn_k, sigma_factor) must be identical across dispatch."""
    winners_serial: dict[str, dict] = {}
    winners_parallel: dict[str, dict] = {}

    run_stage0(
        topologies=('circle',), n_jobs=1,
        capture_winners_into=winners_serial,
        **SMALL_KWARGS,
    )
    run_stage0(
        topologies=('circle',), n_jobs=2,
        capture_winners_into=winners_parallel,
        **SMALL_KWARGS,
    )

    for topo in winners_serial:
        assert (winners_serial[topo]['knn_k'],
                winners_serial[topo]['sigma_factor']) == \
               (winners_parallel[topo]['knn_k'],
                winners_parallel[topo]['sigma_factor'])
