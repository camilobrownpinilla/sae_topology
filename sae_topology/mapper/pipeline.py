"""Mapper pipeline: cover -> cluster -> graph -> Betti, plus the 20-config
hyperparameter sweep.

Wraps kmapper.KeplerMapper with our chosen single-linkage / first-gap
clusterer and computes Betti via networkx + the Euler formula.

Per spec section 4.1, lines 95-127.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import networkx as nx
import scipy.sparse as sp_sparse
import kmapper as km
from kmapper.cover import Cover
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors


# Hyperparameter sweep grid (stage0_tuning.md §3.4 — extended to include
# n_intervals=35 so the full 6×4=24 grid covers the regime needed at N=100k).
N_INTERVALS_GRID = [5, 8, 12, 18, 25, 35]
OVERLAP_GRID = [0.15, 0.25, 0.35, 0.50]


def global_distance_threshold(Z: np.ndarray, knn: int = 15,
                              factor: float = 5.0) -> float:
    """Compute a global single-linkage threshold from the full dataset's kNN
    distances. Within each cover box, points within `factor * median_kNN`
    are merged into one cluster. Distinct components separated by larger
    bridges are split.

    Default `factor=5` is conservative: a uniform manifold sample has
    nearest-neighbor distances at scale d_kNN; bridge edges between truly
    disconnected pieces are at scale O(separation), typically >> 5*d_kNN.
    """
    Z = np.asarray(Z, dtype=float)
    n = len(Z)
    knn = min(knn, n - 1)
    if knn < 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=knn + 1)
    nn.fit(Z)
    distances, _ = nn.kneighbors(Z)
    median_knn = float(np.median(distances[:, 1:]))
    return factor * median_knn


class FirstGapAgglomerative(ClusterMixin, BaseEstimator):
    """Single-linkage clusterer with a global distance threshold.

    Spec section 4.1 line 99 calls for "single-linkage with first-gap
    threshold on the dendrogram." The honest "first gap" version (largest
    or first-empty-bin gap on the per-box dendrogram) over-fragments dense
    manifold data, because single-linkage merge heights on a connected
    sample have a long tail (the well-known chain effect). Here we instead
    use a globally-computed threshold derived from the full dataset's
    nearest-neighbor distances:

        threshold = factor * median(kNN distances in Z)

    This puts every point that's within `factor` nearest-neighbor distances
    of another into the same cluster. Connected manifold patches inside a
    cover box merge to one cluster; bridge edges between genuinely
    disconnected pieces (e.g. the two arcs at a figure-8 wedge if the
    cover doesn't separate them) exceed the threshold and split.
    """

    def __init__(self, distance_threshold: float = 0.0):
        self.distance_threshold = distance_threshold
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n == 0:
            self.labels_ = np.array([], dtype=int)
            return self
        if n == 1:
            self.labels_ = np.array([0], dtype=int)
            return self
        if self.distance_threshold <= 0:
            self.labels_ = np.zeros(n, dtype=int)
            return self
        Z = linkage(X, method='single')
        self.labels_ = fcluster(Z, t=self.distance_threshold, criterion='distance').astype(int) - 1
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


def run_mapper_once(
    Z: np.ndarray,
    filter_values: np.ndarray,
    n_intervals: int,
    overlap: float,
    clusterer=None,
    distance_threshold: Optional[float] = None,
) -> dict:
    """Run kmapper once at the given hyperparameters, return the kmapper graph dict.

    If `clusterer` is None, build a `FirstGapAgglomerative` with the supplied
    `distance_threshold` (or compute one via `global_distance_threshold(Z)`
    when `distance_threshold` is None as well).
    """
    if clusterer is None:
        if distance_threshold is None:
            distance_threshold = global_distance_threshold(np.asarray(Z))
        clusterer = FirstGapAgglomerative(distance_threshold=distance_threshold)
    mapper = km.KeplerMapper(verbose=0)
    cover = Cover(n_cubes=n_intervals, perc_overlap=overlap)
    return mapper.map(
        lens=np.asarray(filter_values),
        X=np.asarray(Z),
        cover=cover,
        clusterer=clusterer,
    )


def graph_betti(graph: dict) -> tuple[int, int]:
    """Return (b_0, b_1) of the Mapper *simplicial complex* (graph 1-skeleton
    plus filled 2-cells where 3 nodes share a common point).

    A bare graph computation (b_1 = E - V + b_0) over-counts cycles whenever
    the cover has triple-overlap regions: 3 cover boxes that all contain the
    same point produce 3 mutually-connected nodes (a triangular subgraph),
    but in the Mapper simplicial complex this triangle is filled (a 2-cell),
    so it does NOT contribute to H_1. We fill those triangles and compute
    Betti via simplicial homology over GF(2) (correct for our torsion-free
    toy manifolds). Per the original Singh-Mémoli-Carlsson construction.
    """
    nodes = list(graph['nodes'].keys())
    nid_to_pts = {nid: frozenset(graph['nodes'][nid]) for nid in nodes}
    nid_to_idx = {nid: i for i, nid in enumerate(nodes)}
    V = len(nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for nid, nbrs in graph.get('links', {}).items():
        for nb in nbrs:
            G.add_edge(nid, nb)
    edges = sorted({tuple(sorted(e)) for e in G.edges()})
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)
    b0 = int(nx.number_connected_components(G))

    if E == 0:
        return b0, 0

    triangles: list[tuple[str, str, str]] = []
    for a, b in edges:
        common_nbrs = set(G.neighbors(a)) & set(G.neighbors(b))
        for c in common_nbrs:
            if nid_to_idx[c] <= max(nid_to_idx[a], nid_to_idx[b]):
                continue
            if nid_to_pts[a] & nid_to_pts[b] & nid_to_pts[c]:
                triangles.append(tuple(sorted([a, b, c])))

    rank2 = 0
    if triangles:
        rows: list[set[int]] = []
        for a, b, c in triangles:
            rows.append({
                edge_to_idx[(a, b)],
                edge_to_idx[(a, c)],
                edge_to_idx[(b, c)],
            })
        rank2 = _gf2_rank(rows)

    rank1 = V - b0
    b1 = int(E - rank1 - rank2)
    return b0, b1


def _gf2_rank(rows: list[set[int]]) -> int:
    """Rank of a binary matrix over GF(2). Each row is a set of column indices
    where the value is 1.

    For boundary matrices of simplicial complexes whose homology is
    torsion-free (true for all our toy manifolds: S^1, T^2, S^2, figure-8),
    GF(2) rank equals integer/rational rank, so b_1 over Q matches b_1 over
    Z/2Z. Gaussian elimination over GF(2) is O(R * C) bit-ops per row;
    using Python `set` xor it's O(R^2 * avg_nnz_per_row) for our sparsities,
    which is fast for R, C up to a few thousand.
    """
    pivots: dict[int, set[int]] = {}
    rank = 0
    for row in rows:
        cur = set(row)
        while cur:
            lead = min(cur)
            if lead in pivots:
                cur ^= pivots[lead]
            else:
                pivots[lead] = cur
                rank += 1
                break
    return rank


def node_means(graph: dict, point_values: np.ndarray) -> np.ndarray:
    """For each node in `graph['nodes']`, compute the mean of `point_values`
    over the node's member point indices.

    `point_values` must be indexable by the integer indices stored in
    `graph['nodes']`. May be 1-D `(N,)` or 2-D `(N, k)`. Empty nodes return NaN.
    """
    point_values = np.asarray(point_values)
    out_shape = (len(graph['nodes']),) if point_values.ndim == 1 \
        else (len(graph['nodes']), point_values.shape[1])
    out = np.full(out_shape, np.nan, dtype=float)
    for i, (_, members) in enumerate(graph['nodes'].items()):
        if not members:
            continue
        out[i] = point_values[members].mean(axis=0)
    return out


def pin_eigvec_sign(v: np.ndarray) -> np.ndarray:
    """Multiply v by +/-1 so v[argmax(|v|)] >= 0. Removes the eigvec sign
    ambiguity that would otherwise flip colorbars across runs / seeds."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        return -v
    return v


_CUBE_PREFIX = re.compile(r"^(cube\d+)")


def node_to_box_ratio(graph: dict) -> float:
    """Total Mapper nodes / number of cover boxes that produced >=1 node.

    Ratio = 1.0 is perfect; > 1.0 indicates fragmentation (a single cover box
    fractured into multiple clusters because the underlying patch tore in
    activation space). Spec section 4.1 line 124.
    """
    node_ids = list(graph['nodes'].keys())
    if not node_ids:
        return float('nan')
    cubes: set[str] = set()
    for nid in node_ids:
        match = _CUBE_PREFIX.match(nid)
        if match:
            cubes.add(match.group(1))
        else:
            cubes.add(nid)
    if not cubes:
        return float('nan')
    return len(node_ids) / len(cubes)


# Expected nerve Betti numbers per (manifold, filter-dim) pair.
# Used by `nerve_mismatch` to compare against the cover-nerve baseline.
EXPECTED_NERVE_BETTI = {
    'circle': (1, 1),
    'torus':  (1, 2),
    'sphere': (1, 0),  # nerve of a triangulation of S^2 is a 2-sphere; only b_0, b_1 reportable on graphs.
    'figure_eight': (1, 2),
    'two_circles': (2, 2),
}


def nerve_mismatch(graph: dict, topology: str) -> dict:
    """Return {b0_obs, b1_obs, b0_expected, b1_expected, mismatch_b0, mismatch_b1}.

    For a regular cover of the eigenvector image (spec section 4.1 line 125),
    the expected Mapper graph is the nerve of that cover: a cycle for S^1, a
    toroidal grid for T^2, a spherical triangulation for S^2.
    """
    b0_obs, b1_obs = graph_betti(graph)
    b0_exp, b1_exp = EXPECTED_NERVE_BETTI.get(topology, (None, None))
    return {
        'b0_obs': b0_obs,
        'b1_obs': b1_obs,
        'b0_expected': b0_exp,
        'b1_expected': b1_exp,
        'mismatch_b0': None if b0_exp is None else b0_obs - b0_exp,
        'mismatch_b1': None if b1_exp is None else b1_obs - b1_exp,
    }


def _mapper_one_config(
    Z: np.ndarray,
    filter_values: np.ndarray,
    ni: int,
    ov: float,
    distance_threshold: float,
    topology: Optional[str],
) -> tuple:
    """Worker function: run Mapper at one (ni, ov) config and return
    `((ni, ov), entry_dict)`. Pinned to a single BLAS thread so that
    parallel and serial paths produce identical numerical results."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        graph = run_mapper_once(
            Z, filter_values, ni, ov,
            distance_threshold=distance_threshold,
        )
        b0, b1 = graph_betti(graph)
        n2b = node_to_box_ratio(graph)
        entry = {
            'b0': b0,
            'b1': b1,
            'node_to_box': n2b,
            'n_nodes': len(graph['nodes']),
            'n_edges': sum(len(v) for v in graph.get('links', {}).values()) // 2,
        }
        if topology is not None:
            entry['nerve_mismatch'] = nerve_mismatch(graph, topology)
    return (ni, ov), entry


def mapper_sweep(
    Z: np.ndarray,
    filter_values: np.ndarray,
    topology: Optional[str] = None,
    n_intervals_grid: Iterable[int] = N_INTERVALS_GRID,
    overlap_grid: Iterable[float] = OVERLAP_GRID,
    distance_threshold: Optional[float] = None,
    n_jobs: int = 1,
) -> dict:
    """Run the 20-config hyperparameter sweep and return per-config diagnostics.

    The single-linkage `distance_threshold` is computed once globally from
    Z (default) and reused across all 20 configs, so the only varying
    hyperparameters are the cover (`n_intervals`, `overlap`).

    Args:
        n_jobs:  if >1, dispatch the (n_intervals, overlap) pairs across
                 `n_jobs` joblib workers (loky backend, BLAS pinned to 1
                 thread per worker for determinism). Default 1 (serial).
                 Importers calling mapper_sweep from inside an outer joblib
                 pool MUST pass n_jobs=1 (no nested loky pools).

    Returns dict keyed by (n_intervals, overlap) tuple. Each value:
        b0, b1, node_to_box, nerve_mismatch (optional), n_nodes, n_edges.
    """
    if distance_threshold is None:
        distance_threshold = global_distance_threshold(np.asarray(Z))

    pairs = [(int(ni), float(ov))
             for ni in n_intervals_grid for ov in overlap_grid]

    if n_jobs == 1 or len(pairs) <= 1:
        results: dict = {}
        for ni, ov in pairs:
            key, entry = _mapper_one_config(
                Z, filter_values, ni, ov, distance_threshold, topology,
            )
            results[key] = entry
        return results

    from joblib import Parallel, delayed
    pairs_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_mapper_one_config)(
            Z, filter_values, ni, ov, distance_threshold, topology,
        )
        for ni, ov in pairs
    )
    # Re-key into a dict; preserve the canonical ordering (sorted by pairs
    # list, which is the same as the serial-loop order) so downstream
    # consumers see identical iteration order.
    out: dict = {}
    for key, entry in pairs_results:
        out[key] = entry
    return out


def stable_betti(sweep_results: dict) -> tuple[int, int, float]:
    """Most-frequent (b_0, b_1) pair across the sweep, with its share.

    Useful as a quick informal summary, but does NOT enforce contiguity in
    the (n_intervals, overlap) grid and does not check correctness against
    a known ground truth. For Stage-0 / Stage-2 headlines, prefer
    `correct_region` (which requires both contiguity AND match-to-GT).
    """
    counts: dict = {}
    for entry in sweep_results.values():
        key = (entry['b0'], entry['b1'])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return (0, 0, 0.0)
    best, best_count = max(counts.items(), key=lambda kv: kv[1])
    return (best[0], best[1], best_count / len(sweep_results))


def failure_mode_diagnostic(
    sweep_results: dict,
    expected_betti: tuple[int, int],
) -> dict:
    """Per stage0_tuning.md §4.1.3: configurations whose Betti is wrong should
    fail in *interpretable* directions. b_1 should be monotone non-decreasing
    in n_intervals at each fixed overlap (coarse covers miss cycles, fine
    covers gain spurious ones from sampling noise).

    Returns:
        rows_monotone:    list of {overlap, b1_by_ni: list[int],
                                   monotone_nondecreasing: bool} for each
                                   overlap value.
        all_rows_monotone: bool — True iff every fixed-overlap row's b_1 is
                                  non-decreasing in n_intervals.
        b1_expected:      expected b_1 (for context).

    This is a qualitative warning, not a hard pass/fail; the spec says
    "If failures are erratic — e.g., b_1=5 at coarse and b_1=0 at fine — the
    pipeline is broken even when the stable region looks correct."
    """
    if not sweep_results:
        return {'rows_monotone': [], 'all_rows_monotone': True,
                'b1_expected': expected_betti[1]}
    ni_values = sorted({k[0] for k in sweep_results})
    ov_values = sorted({k[1] for k in sweep_results})

    rows = []
    all_mono = True
    for ov in ov_values:
        b1_row = []
        for ni in ni_values:
            entry = sweep_results.get((ni, ov))
            if entry is None:
                b1_row.append(None)
            else:
                b1_row.append(int(entry['b1']))
        defined = [v for v in b1_row if v is not None]
        mono = all(defined[i] <= defined[i + 1] for i in range(len(defined) - 1))
        rows.append({
            'overlap': float(ov),
            'b1_by_ni': b1_row,
            'monotone_nondecreasing': bool(mono),
        })
        all_mono = all_mono and mono
    return {
        'rows_monotone': rows,
        'all_rows_monotone': bool(all_mono),
        'b1_expected': int(expected_betti[1]),
    }


def correct_region(
    sweep_results: dict,
    expected_betti: tuple[int, int],
) -> dict:
    """Largest 4-connected region in the (n_intervals, overlap) grid where
    Betti exactly equals `expected_betti`.

    This is the methodologically honest version of "stable region" for
    Stage-0 validation (where ground-truth Betti is known): a modal pair
    that's wrong is not stability, and a correct pair that's scattered
    is weak evidence. Reporting the largest contiguous correct block
    measures both correctness and robustness to cover-hyperparameter
    perturbation.

    Returns:
        region_size:        cells in the largest contiguous correct block
        region_fraction:    region_size / total grid cells
        n_correct_total:    correct configs anywhere in the grid (modal-
                            style; reported alongside for diagnosis)
        total_configs:      grid size (typically 24)
        is_interior:        True iff the largest contiguous correct block
                            does not touch any of the four grid edges
                            (per stage0_tuning.md §3.4: a stable region at
                            the grid boundary may extend further outside
                            the swept range, indicating the grid is too
                            narrow).
    """
    if not sweep_results:
        return {'region_size': 0, 'region_fraction': 0.0,
                'n_correct_total': 0, 'total_configs': 0,
                'is_interior': False}

    ni_values = sorted({k[0] for k in sweep_results})
    ov_values = sorted({k[1] for k in sweep_results})
    ni_idx = {v: i for i, v in enumerate(ni_values)}
    ov_idx = {v: i for i, v in enumerate(ov_values)}
    ni_max_idx = len(ni_values) - 1
    ov_max_idx = len(ov_values) - 1

    correct: set[tuple[int, int]] = set()
    for (ni, ov), v in sweep_results.items():
        if (v['b0'], v['b1']) == tuple(expected_betti):
            correct.add((ni_idx[ni], ov_idx[ov]))

    visited: set[tuple[int, int]] = set()
    best_size = 0
    best_cells: set[tuple[int, int]] = set()
    for start in correct:
        if start in visited:
            continue
        size = 0
        cells: set[tuple[int, int]] = set()
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in visited or cell not in correct:
                continue
            visited.add(cell)
            cells.add(cell)
            size += 1
            i, j = cell
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                stack.append((i + di, j + dj))
        if size > best_size:
            best_size = size
            best_cells = cells

    is_interior = bool(best_cells) and not any(
        i == 0 or i == ni_max_idx or j == 0 or j == ov_max_idx
        for (i, j) in best_cells
    )

    total = len(sweep_results)
    return {
        'region_size': best_size,
        'region_fraction': best_size / total,
        'n_correct_total': len(correct),
        'total_configs': total,
        'is_interior': is_interior,
    }
