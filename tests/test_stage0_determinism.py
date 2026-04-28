"""End-to-end determinism test for the Stage 0 flat-parallel pipeline.

Asserts that `run_stage0` produces bit-identical Stage0Results regardless of
`n_jobs`. Runs at small N to keep the test fast (~30 s); the same invariants
hold at full N.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sae_topology.experiments.stage0_validate import (
    Stage0Result,
    run_stage0,
)


# Compact knobs for a fast, deterministic test.
SMALL_KWARGS = dict(
    ambient_d=8,
    sigma=0.05,
    seed=0,
    knn_k_grid=(15, 25),                         # 2 knn_k -> 2 Stage A jobs / topo
    sigma_factor_grid=(0.5, 1.0),                 # 2 sigma_factor inner iterations
    n_intervals_grid=(5, 8),                      # 4-config Stage B grid
    overlap_grid=(0.25, 0.50),
    inner_n_intervals_grid=(5, 8),                # 4-config inner grid
    spectral_K=10,
    n_samples_override={'circle': 600, 'torus': 600},
)


def _entry_eq(a: dict, b: dict) -> None:
    """Compare two `tuning_log` entries (no eigenvectors stored here)."""
    assert set(a.keys()) == set(b.keys()), f'keys differ: {a.keys()} vs {b.keys()}'
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, dict):
            assert va == vb, f'{k}: dict mismatch'
        elif isinstance(va, list):  # spectral_eigenvalues
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
        elif isinstance(va, float):
            if np.isnan(va):
                assert np.isnan(vb)
            else:
                assert va == vb, f'{k}: {va} != {vb}'
        else:
            assert va == vb, f'{k}: {va} != {vb}'


def _result_eq(a: Stage0Result, b: Stage0Result) -> None:
    """Compare two Stage0Results for one topology."""
    da = dataclasses.asdict(a)
    db = dataclasses.asdict(b)
    # Tuning logs are large; compare entry-wise so failures are localised.
    log_a = da.pop('tuning_log')
    log_b = db.pop('tuning_log')
    assert len(log_a) == len(log_b), f'tuning_log length differs'
    for i, (ea, eb) in enumerate(zip(log_a, log_b)):
        _entry_eq(ea, eb)
    # Compare scalar / dict / tuple fields.
    for key in da:
        va, vb = da[key], db[key]
        if isinstance(va, list):
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
        elif isinstance(va, float) and np.isnan(va):
            assert np.isnan(vb), f'{key}: NaN vs {vb}'
        else:
            assert va == vb, f'{key}: {va!r} vs {vb!r}'


@pytest.mark.parametrize('n_jobs', [2, 4])
def test_stage0_serial_equals_parallel(n_jobs: int):
    """Stage 0 must produce identical results in serial vs parallel paths."""
    serial = run_stage0(
        topologies=('circle', 'torus'), n_jobs=1, **SMALL_KWARGS,
    )
    parallel = run_stage0(
        topologies=('circle', 'torus'), n_jobs=n_jobs, **SMALL_KWARGS,
    )
    assert set(serial.keys()) == set(parallel.keys())
    for topo in serial:
        _result_eq(serial[topo], parallel[topo])


def test_stage0_winner_is_stable_across_dispatch():
    """A weaker but more focused check: the (knn_k, sigma_factor) winner per
    topology must be identical across dispatch paths."""
    serial = run_stage0(
        topologies=('circle', 'torus'), n_jobs=1, **SMALL_KWARGS,
    )
    parallel = run_stage0(
        topologies=('circle', 'torus'), n_jobs=4, **SMALL_KWARGS,
    )
    for topo in serial:
        assert (serial[topo].knn_k, serial[topo].sigma_factor) == \
               (parallel[topo].knn_k, parallel[topo].sigma_factor), (
                   f'{topo}: winner differs'
               )


def test_stage0_yaml_output_identical(tmp_path):
    """The baseline.yaml file written to disk must be byte-identical between
    serial and parallel paths."""
    out_serial = tmp_path / 'serial.yaml'
    out_parallel = tmp_path / 'parallel.yaml'
    run_stage0(
        topologies=('circle',), n_jobs=1,
        out_yaml=out_serial, out_json=None,
        **{k: v for k, v in SMALL_KWARGS.items() if k != 'n_samples_override'},
        n_samples_override={'circle': 600},
    )
    run_stage0(
        topologies=('circle',), n_jobs=2,
        out_yaml=out_parallel, out_json=None,
        **{k: v for k, v in SMALL_KWARGS.items() if k != 'n_samples_override'},
        n_samples_override={'circle': 600},
    )
    assert out_serial.read_bytes() == out_parallel.read_bytes()
