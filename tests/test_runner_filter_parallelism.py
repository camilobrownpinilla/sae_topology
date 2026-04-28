"""Determinism test for `_run_mapper_all_filters` — the per-filter Mapper
sweep dispatch in `sae_topology.experiments.runner`.

The runner uses joblib to dispatch the (Laplacian, ground-truth, PCA) filter
sweeps in parallel when `n_jobs > 1`. We assert that serial and parallel
paths produce identical per-filter `_summary` payloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.experiments.runner import _run_mapper_all_filters


def _summaries_equal(a: dict, b: dict) -> None:
    """Assert two `_run_mapper_all_filters` outputs are equal entry-wise."""
    assert set(a.keys()) == set(b.keys()), (
        f'filter sets differ: {set(a.keys()) ^ set(b.keys())}'
    )
    # Order check: the canonical (lap, gt?, pca) ordering must be preserved.
    assert list(a.keys()) == list(b.keys()), (
        f'filter ordering differs: {list(a.keys())} vs {list(b.keys())}'
    )
    for filter_kind in a:
        sa, sb = a[filter_kind], b[filter_kind]
        # The per-filter payload is a dict of (cover-config-key -> entry-dict)
        # plus a '_summary' key. Compare entries one-by-one.
        assert set(sa.keys()) == set(sb.keys()), (
            f'{filter_kind}: cover-key sets differ'
        )
        for k in sa:
            ea, eb = sa[k], sb[k]
            if isinstance(ea, dict):
                # cover-config dict OR _summary dict
                assert ea.keys() == eb.keys(), (
                    f'{filter_kind}/{k}: subkey sets differ'
                )
                for kk in ea:
                    va, vb = ea[kk], eb[kk]
                    if isinstance(va, dict):
                        assert va == vb, f'{filter_kind}/{k}/{kk}: dict mismatch'
                    elif isinstance(va, list):
                        np.testing.assert_array_equal(
                            np.asarray(va), np.asarray(vb),
                        )
                    elif isinstance(va, float):
                        if np.isnan(va):
                            assert np.isnan(vb)
                        else:
                            assert va == vb, (
                                f'{filter_kind}/{k}/{kk}: {va} != {vb}'
                            )
                    else:
                        assert va == vb, (
                            f'{filter_kind}/{k}/{kk}: {va!r} vs {vb!r}'
                        )
            else:
                assert ea == eb, f'{filter_kind}/{k}: {ea!r} vs {eb!r}'


@pytest.mark.parametrize('n_jobs', [2, 3])
def test_runner_filters_serial_equals_parallel_circle(circle_small, n_jobs):
    """Circle: lap + gt + pca = 3 filters, can saturate n_jobs in {2, 3}."""
    X, gt = circle_small
    serial = _run_mapper_all_filters(
        X, gt, 'circle', knn_k=15, sigma_factor=1.0, n_jobs=1,
    )
    parallel = _run_mapper_all_filters(
        X, gt, 'circle', knn_k=15, sigma_factor=1.0, n_jobs=n_jobs,
    )
    _summaries_equal(serial, parallel)


def test_runner_filters_serial_equals_parallel_torus(torus_small):
    X, gt = torus_small
    serial = _run_mapper_all_filters(
        X, gt, 'torus', knn_k=15, sigma_factor=1.0, n_jobs=1,
    )
    parallel = _run_mapper_all_filters(
        X, gt, 'torus', knn_k=15, sigma_factor=1.0, n_jobs=2,
    )
    _summaries_equal(serial, parallel)


def test_runner_filters_no_gt_skips_ground_truth_filter():
    """A topology without a closed-form GT filter (e.g. figure_eight) must
    yield 2 filters, not 3 — and the ordering still matches."""
    from sae_topology.dgp import make_dgp
    np.random.seed(0)
    dgp = make_dgp('figure_eight', d=8, sigma=0.05, c0=np.zeros(8))
    X, gt = dgp.sample_with_gt(500)

    out_serial = _run_mapper_all_filters(
        X, gt, 'figure_eight', knn_k=15, sigma_factor=1.0, n_jobs=1,
    )
    out_parallel = _run_mapper_all_filters(
        X, gt, 'figure_eight', knn_k=15, sigma_factor=1.0, n_jobs=2,
    )
    assert list(out_serial.keys()) == ['laplacian', 'pca']
    _summaries_equal(out_serial, out_parallel)
