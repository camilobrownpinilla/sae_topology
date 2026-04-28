"""Smoke tests for Stage 0 artifact persistence + plotting helpers.

Tiny circle fixture (N=200) so each test runs in <2s. We don't assert
pixel content — just that files are written and round-trip correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sae_topology.experiments.stage0_artifacts import (
    persist_topology_bundle,
    render_signoff_md,
)
from sae_topology.experiments.stage0_plots import (
    mapper_grid_heatmap,
    spectrum_overlay_plot,
    mapper_graph_renderings,
    stable_configs_from_correct_region,
)
from sae_topology.experiments.stage0_validate import Stage0Result


# ---------------------------------------------------------------------------
# persist_topology_bundle round-trip
# ---------------------------------------------------------------------------

def test_persist_topology_bundle_writes_all_files(tmp_path):
    N, K, k_intrinsic, ambient_d = 100, 10, 2, 8
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, ambient_d))
    gt = rng.standard_normal((N, k_intrinsic))
    eigvals = np.linspace(0, 1, K)
    eigvecs = rng.standard_normal((N, K))

    sweep_lap = {(5, 0.15): {'b0': 1, 'b1': 1, 'n_nodes': 5,
                              'n_edges': 5, 'node_to_box': 1.0}}

    out_dir = persist_topology_bundle(
        topology='circle', out_dir=tmp_path,
        X=X, gt=gt,
        eigenvalues=eigvals, eigenvectors=eigvecs,
        sigma_used=0.1, knn_k=15, sigma_factor=1.0, seed=0,
        sweep_lap=sweep_lap,
    )

    assert (out_dir / 'samples.npz').exists()
    assert (out_dir / 'spectrum.npz').exists()
    assert (out_dir / 'mapper_sweep.json').exists()
    assert (out_dir / 'report.json').exists()

    # Roundtrip samples + spectrum
    samples = np.load(out_dir / 'samples.npz')
    np.testing.assert_array_equal(samples['X'], X)
    np.testing.assert_array_equal(samples['gt'], gt)
    assert int(samples['seed']) == 0

    spec = np.load(out_dir / 'spectrum.npz')
    np.testing.assert_allclose(spec['eigenvalues'], eigvals)
    np.testing.assert_allclose(spec['eigenvectors'], eigvecs)
    assert int(spec['knn_k']) == 15

    # mapper_sweep.json is JSON-serialisable and has lap entries
    with open(out_dir / 'mapper_sweep.json') as f:
        data = json.load(f)
    assert 'laplacian' in data
    assert len(data['laplacian']) == 1


# ---------------------------------------------------------------------------
# render_signoff_md
# ---------------------------------------------------------------------------

def _fake_stage0_result() -> Stage0Result:
    return Stage0Result(
        topology='circle',
        n_samples=1000,
        ambient_d=8,
        sigma_noise=0.05,
        knn_k=25,
        sigma_factor=1.0,
        sigma_used=0.15,
        spectral_log_ratio_error=0.03,
        spectral_eigenvalues=[0.0, 1.0, 1.0, 4.0, 4.0],
        mapper_laplacian_correct_region={
            'region_size': 13, 'region_fraction': 0.65,
            'n_correct_total': 13, 'total_configs': 24, 'is_interior': True,
        },
        mapper_gt_correct_region={
            'region_size': 16, 'region_fraction': 0.80,
            'n_correct_total': 16, 'total_configs': 24, 'is_interior': True,
        },
        mapper_pca_correct_region={
            'region_size': 10, 'region_fraction': 0.50,
            'n_correct_total': 10, 'total_configs': 24, 'is_interior': True,
        },
        mapper_laplacian_modal_betti=(1, 1),
        spectral_pass=True,
        mapper_pass=True,
        overall_pass=True,
        tuning_log=[],
        multiplicity_pass=True,
        multiplicity_check={'all_match': True, 'per_cluster': []},
        near_zero_pass=True,
        near_zero_count=1,
        mapper_interior_pass=True,
        failure_mode_diagnostic={'all_rows_monotone': True},
    )


def test_render_signoff_md_circle_pass():
    md = render_signoff_md(
        topology='circle',
        stage0_result=_fake_stage0_result(),
        reproducibility=None,
        n_clusters=4,
    )
    assert '`circle`' in md
    assert 'PASS' in md
    # All-x checklist for a passing circle
    assert md.count('- [x]') >= 8


def test_render_signoff_md_torus_with_repro():
    r = _fake_stage0_result()
    md = render_signoff_md(
        topology='torus',
        stage0_result=r,
        reproducibility={'pass': True, 'log_ratio_error_diff': 0.005},
        n_clusters=4,
    )
    assert 'torus' in md
    assert 'E_diff=0.0050' in md or 'E_diff=0.005' in md


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def test_mapper_grid_heatmap_writes_file(tmp_path):
    sweep = {(5, 0.15): {'b0': 1, 'b1': 1},
             (5, 0.25): {'b0': 1, 'b1': 0},
             (8, 0.15): {'b0': 1, 'b1': 1},
             (8, 0.25): {'b0': 1, 'b1': 1}}
    out = mapper_grid_heatmap(sweep, (1, 1), tmp_path / 'h.png')
    assert out.exists()
    assert out.stat().st_size > 1000  # non-empty PNG


def test_spectrum_overlay_writes_file(tmp_path):
    emp = np.array([0.0, 1.0, 1.0, 4.0, 4.0])
    theory = [0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0]
    mc = {'per_cluster': [{'level': 1.0, 'theory_mult': 2,
                           'emp_count': 2, 'match': True}]}
    out = spectrum_overlay_plot(emp, theory, mc, tmp_path / 's.png')
    assert out.exists()


def test_stable_configs_returns_correct_cells():
    sweep = {}
    for ni in [5, 8, 12, 18, 25, 35]:
        for ov in [0.15, 0.25, 0.35, 0.50]:
            if ni in [8, 12, 18] and ov in [0.25, 0.35]:
                sweep[(ni, ov)] = {'b0': 1, 'b1': 1}
            else:
                sweep[(ni, ov)] = {'b0': 1, 'b1': 0}
    cells = stable_configs_from_correct_region(sweep, (1, 1))
    assert set(cells) == {(8, 0.25), (8, 0.35), (12, 0.25), (12, 0.35),
                          (18, 0.25), (18, 0.35)}


def test_mapper_graph_renderings_circle_smoke(tmp_path):
    """Tiny circle: build a real Mapper graph and render it via the helper."""
    from sae_topology.dgp import make_dgp
    from sae_topology.mapper import (
        run_mapper_once, global_distance_threshold,
        laplacian_eigenvector_filter,
    )
    from sae_topology.spectral import coifman_lafon_spectrum

    np.random.seed(0)
    dgp = make_dgp('circle', d=8, sigma=0.05, c0=np.zeros(8))
    X, gt = dgp.sample_with_gt(200)
    spec = coifman_lafon_spectrum(X, knn_k=15, K=4)
    lens = laplacian_eigenvector_filter(X, k=2, eigenvectors=spec['eigenvectors'])
    thresh = float(global_distance_threshold(X))
    graph = run_mapper_once(X, lens, 8, 0.35, distance_threshold=thresh)

    out_dir = tmp_path / 'renderings'
    paths = mapper_graph_renderings(
        graph_by_config={(8, 0.35): graph},
        gt=gt,
        topology='circle',
        stable_configs=[(8, 0.35)],
        out_dir=out_dir,
    )
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].stat().st_size > 1000
