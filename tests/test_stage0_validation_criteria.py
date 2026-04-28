"""Unit tests for the Stage 0 validation criteria added in this revision:

  - multiplicity_clusters_match (sae_topology.spectral.metrics)
  - correct_region's `is_interior` flag (sae_topology.mapper.pipeline)
  - failure_mode_diagnostic (sae_topology.mapper.pipeline)

Each is exercised on small synthetic fixtures so the tests run in <1s.
"""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.mapper.pipeline import correct_region, failure_mode_diagnostic
from sae_topology.spectral.metrics import multiplicity_clusters_match
from sae_topology.spectral.reference import (
    S1_LEVELS, S1_MULTS,
    T2_LEVELS, T2_MULTS,
    S2_LEVELS, S2_MULTS,
)


# ---------------------------------------------------------------------------
# multiplicity_clusters_match
# ---------------------------------------------------------------------------

def test_multiplicity_clusters_match_s1_perfect():
    # First 3 non-trivial S^1 levels: ratio 1, 4, 9 with mult 2 each.
    emp = np.array([0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0])
    out = multiplicity_clusters_match(emp, S1_LEVELS, S1_MULTS, n_clusters=3)
    assert out['all_match']
    assert out['n_clusters_checked'] == 3
    for c in out['per_cluster']:
        assert c['emp_count'] == c['theory_mult']


def test_multiplicity_clusters_match_t2_perfect():
    # First 4 non-trivial T^2 levels: ratios 1, 2, 4, 5 with mult 4, 4, 4, 8.
    emp = np.array([0.0] + [1.0] * 4 + [2.0] * 4 + [4.0] * 4 + [5.0] * 8)
    out = multiplicity_clusters_match(emp, T2_LEVELS, T2_MULTS, n_clusters=4)
    assert out['all_match']


def test_multiplicity_clusters_match_s2_perfect():
    # First 4 non-trivial S^2 levels (after dividing by lambda_1=2):
    # 1, 3, 6, 10 with mult 3, 5, 7, 9.
    emp = np.array([0.0] + [1.0] * 3 + [3.0] * 5 + [6.0] * 7 + [10.0] * 9)
    out = multiplicity_clusters_match(emp, S2_LEVELS, S2_MULTS, n_clusters=4)
    assert out['all_match']


def test_multiplicity_clusters_match_off_by_one_count_fails():
    # S^1 with one missing eigenvalue at the lambda_2 = 4 level.
    emp = np.array([0.0, 1.0, 1.0, 4.0, 9.0, 9.0])  # 1 instead of 2 at level 4
    out = multiplicity_clusters_match(emp, S1_LEVELS, S1_MULTS, n_clusters=2)
    assert not out['all_match']
    # The level-1 cluster matches, level-4 does not.
    matches = [c['match'] for c in out['per_cluster']]
    assert matches == [True, False]


def test_multiplicity_clusters_match_eps_window():
    # lambda_1 anchored at 1.0 so normalisation is a no-op; the second
    # eig is 0.04 away from level 1 and the level-4 cluster is at 3.96/4.04.
    # eps=0.05 should pass (|0.04|<0.05); eps=0.02 should fail.
    emp = np.array([0.0, 1.0, 1.04, 3.96, 4.04])
    eps_loose = multiplicity_clusters_match(
        emp, S1_LEVELS, S1_MULTS, n_clusters=2, eps=0.05,
    )
    assert eps_loose['all_match']
    eps_tight = multiplicity_clusters_match(
        emp, S1_LEVELS, S1_MULTS, n_clusters=2, eps=0.02,
    )
    assert not eps_tight['all_match']


def test_multiplicity_clusters_match_all_zero_raises():
    with pytest.raises(ValueError):
        multiplicity_clusters_match(
            np.zeros(5), S1_LEVELS, S1_MULTS, n_clusters=2,
        )


# ---------------------------------------------------------------------------
# correct_region: is_interior flag
# ---------------------------------------------------------------------------

GRID_NI = [5, 8, 12, 18, 25, 35]
GRID_OV = [0.15, 0.25, 0.35, 0.50]


def _make_sweep(correct_cells: set[tuple[int, float]]) -> dict:
    sweep = {}
    for ni in GRID_NI:
        for ov in GRID_OV:
            if (ni, ov) in correct_cells:
                sweep[(ni, ov)] = {'b0': 1, 'b1': 1, 'n_nodes': 5, 'n_edges': 5}
            else:
                sweep[(ni, ov)] = {'b0': 1, 'b1': 0, 'n_nodes': 3, 'n_edges': 2}
    return sweep


def test_correct_region_interior():
    # Interior 4x2 block: ni in {8,12,18,25}, ov in {0.25,0.35}.
    correct = {(ni, ov) for ni in [8, 12, 18, 25] for ov in [0.25, 0.35]}
    cr = correct_region(_make_sweep(correct), (1, 1))
    assert cr['region_size'] == 8
    assert cr['is_interior'] is True


def test_correct_region_touches_left_edge():
    # Touching ni=5 (left edge).
    correct = {(ni, ov) for ni in [5, 8, 12] for ov in [0.25, 0.35]}
    cr = correct_region(_make_sweep(correct), (1, 1))
    assert cr['region_size'] == 6
    assert cr['is_interior'] is False


def test_correct_region_touches_right_edge():
    # Touching ni=35 (right edge).
    correct = {(ni, ov) for ni in [25, 35] for ov in [0.25, 0.35]}
    cr = correct_region(_make_sweep(correct), (1, 1))
    assert cr['region_size'] == 4
    assert cr['is_interior'] is False


def test_correct_region_touches_bottom_edge():
    # Touching ov=0.15 (bottom edge).
    correct = {(ni, ov) for ni in [8, 12, 18] for ov in [0.15, 0.25]}
    cr = correct_region(_make_sweep(correct), (1, 1))
    assert cr['region_size'] == 6
    assert cr['is_interior'] is False


def test_correct_region_touches_top_edge():
    # Touching ov=0.50 (top edge).
    correct = {(ni, ov) for ni in [8, 12, 18] for ov in [0.35, 0.50]}
    cr = correct_region(_make_sweep(correct), (1, 1))
    assert cr['region_size'] == 6
    assert cr['is_interior'] is False


def test_correct_region_empty_returns_not_interior():
    cr = correct_region({}, (1, 1))
    assert cr['region_size'] == 0
    assert cr['is_interior'] is False


def test_correct_region_no_correct_cells_returns_not_interior():
    cr = correct_region(_make_sweep(set()), (1, 1))
    assert cr['region_size'] == 0
    assert cr['is_interior'] is False


# ---------------------------------------------------------------------------
# failure_mode_diagnostic
# ---------------------------------------------------------------------------

def test_failure_mode_monotone_pass():
    # b1 monotone non-decreasing in n_intervals at every overlap row:
    # at each ov: b1 = 0, 0, 1, 1, 2, 2.
    sweep = {}
    for ni_idx, ni in enumerate(GRID_NI):
        for ov in GRID_OV:
            sweep[(ni, ov)] = {'b0': 1, 'b1': ni_idx // 2, 'n_nodes': 3, 'n_edges': 2}
    diag = failure_mode_diagnostic(sweep, (1, 1))
    assert diag['all_rows_monotone'] is True


def test_failure_mode_non_monotone_flagged():
    # Erratic at one row: b1 jumps 5 → 0 in n_intervals direction.
    sweep = {}
    for ni in GRID_NI:
        for ov in GRID_OV:
            if ov == 0.25 and ni == 18:
                b1 = 0  # break monotonicity
            elif ov == 0.25 and ni == 12:
                b1 = 5  # high b1 at moderate ni
            else:
                b1 = 1
            sweep[(ni, ov)] = {'b0': 1, 'b1': b1, 'n_nodes': 3, 'n_edges': 2}
    diag = failure_mode_diagnostic(sweep, (1, 1))
    assert diag['all_rows_monotone'] is False


def test_failure_mode_empty_sweep_returns_default():
    diag = failure_mode_diagnostic({}, (1, 1))
    assert diag['all_rows_monotone'] is True
    assert diag['rows_monotone'] == []
