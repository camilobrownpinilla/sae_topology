"""Stage 0 validation: run the full Mapper + Coifman-Lafon pipeline on RAW
samples (no SAE) for each smooth manifold, AUTO-TUNE (knn_k, sigma_factor)
per manifold, and write the winning config to configs/stage0/baseline.yaml.

Per spec section 7.1.

The methodologically honest version of "stable region" used here is the
largest 4-connected region of the (n_intervals, overlap) grid where Mapper
Betti exactly matches the manifold's known ground truth - see
`sae_topology.mapper.correct_region`. The auto-tune loop picks the
(knn_k, sigma_factor) that maximises this fraction, tie-breaking on the
spectral log-ratio error.

Parallelism (`--n_jobs`, default 1):
  Two flat job pools dispatched via joblib (loky backend, BLAS pinned to
  1 thread per worker for determinism):

    Stage A:  one job per (topology, knn_k); within each job the 3
              sigma_factor variants run serially and share the kNN graph
              (kNN is the cost-dominant part for large N).
              4 topologies x 3 knn_k = 12 jobs.
    Stage B:  one job per (topology, filter_kind); each job runs the full
              (n_intervals x overlap) Mapper sweep on the chosen filter at
              the winning (knn_k, sigma_factor).
              4 topologies x 3 filters - (figure_eight has no GT filter) = 11 jobs.

  Workers MUST call mapper_sweep with n_jobs=1 (no nested loky pools).
  Determinism: parallel and serial paths produce bit-identical baseline.yaml.

Run:  python -m sae_topology.experiments.stage0_validate
      python -m sae_topology.experiments.stage0_validate --n_jobs 12
      python -m sae_topology.experiments.stage0_validate --quick   (N/2 for dev)

Writes:  configs/stage0/baseline.yaml
         configs/stage0/baseline.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from sae_topology.dgp import make_dgp
from sae_topology.mapper import (
    laplacian_eigenvector_filter,
    ground_truth_filter,
    pca_filter,
    mapper_sweep,
    correct_region,
    stable_betti,
    N_INTERVALS_GRID,
    OVERLAP_GRID,
)
from sae_topology.mapper.pipeline import (
    _mapper_one_config,
    global_distance_threshold,
)
from sae_topology.spectral import (
    coifman_lafon_spectrum,
    compute_knn,
    log_ratio_error,
    REFERENCE_SPECTRA,
)


DEFAULT_SAMPLE_SIZES = {
    'circle': 100_000,
    'sphere': 100_000,
    'torus':  100_000,
    'figure_eight': 100_000,
}

# Default topologies for the Stage 0 CLI: only the three smooth manifolds
# in stage0_tuning.md §2. figure_eight remains callable via --topologies.
DEFAULT_STAGE0_TOPOLOGIES = ('circle', 'torus', 'sphere')

# Auto-tune grid (per spec section 7.1 plus our extension).
DEFAULT_KNN_K_GRID = (15, 25, 40)
DEFAULT_SIGMA_FACTOR_GRID = (0.5, 1.0, 2.0)

# Filter dim per topology = multiplicity of the first non-trivial Laplace-
# Beltrami eigenvalue (stage0_tuning.md §4.1):
#   S^1 : λ_1 = 1  has mult 2 (cos θ, sin θ)
#   T^2 : λ_1 = 1  has mult 4 (cos θ, sin θ, cos φ, sin φ)
#   S^2 : λ_1 = 2  has mult 3 (degree-1 spherical harmonics: x, y, z)
FILTER_K_BY_TOPOLOGY = {
    'circle': 2,
    'torus':  4,
    'sphere': 3,
    'figure_eight': 3,
}

# Expected Mapper Betti per manifold (Mapper sees the 1-skeleton + 2-cells,
# so b_2 is not reportable - we compare (b_0, b_1) only).
EXPECTED_BETTI = {
    'circle': (1, 1),
    'torus':  (1, 2),
    'sphere': (1, 0),
    'figure_eight': (1, 2),
}

# Filters supported per topology (figure_eight has no closed-form GT).
GT_FILTER_TOPOLOGIES = {'circle', 'torus', 'sphere'}

# Inner-loop reduced grid (used for cheap scoring during auto-tune).
INNER_NI_GRID = (5, 8, 12)


def _dgp_kwargs(topology: str) -> dict:
    if topology == 'torus':
        return {'major_radius': 1.0, 'minor_radius': 1.0}
    if topology == 'sphere':
        return {'radius': 1.0}
    return {}


def _sample_X_gt(
    topology: str, n_samples: int, ambient_d: int,
    sigma_noise: float, seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deterministic sample. Identical across processes for fixed seed."""
    np.random.seed(seed)
    dgp = make_dgp(
        topology, d=ambient_d, sigma=sigma_noise,
        c0=np.zeros(ambient_d), **_dgp_kwargs(topology),
    )
    return dgp.sample_with_gt(n_samples)


@dataclass
class Stage0Result:
    topology: str
    n_samples: int
    ambient_d: int
    sigma_noise: float

    knn_k: int
    sigma_factor: float
    sigma_used: float

    spectral_log_ratio_error: float | None
    spectral_eigenvalues: list

    mapper_laplacian_correct_region: dict
    mapper_gt_correct_region: dict | None
    mapper_pca_correct_region: dict
    mapper_laplacian_modal_betti: tuple

    spectral_pass: bool
    mapper_pass: bool
    overall_pass: bool

    tuning_log: list  # one entry per (knn_k, sigma_factor) tried

    # Validation criteria (stage0_tuning.md §4 + sign-off checklist §10).
    multiplicity_pass: bool = False
    multiplicity_check: dict | None = None
    near_zero_pass: bool = False
    near_zero_count: int = 0
    mapper_interior_pass: bool = False
    failure_mode_diagnostic: dict | None = None


# ---------------------------------------------------------------------------
# Inner-loop scoring (Stage A)
# ---------------------------------------------------------------------------

def _score_inner(
    X: np.ndarray, topology: str, knn_k: int, sigma_factor: float,
    spec: dict, inner_n_intervals_grid, overlap_grid,
) -> dict:
    """Cheap per-(knn_k, sigma_factor) score: spectral E + Laplacian-filter
    correct-region on the reduced (n_intervals, overlap) grid.
    """
    eigs = spec['eigenvalues']
    has_ref = topology in REFERENCE_SPECTRA
    log_err = None
    if has_ref:
        try:
            log_err = float(log_ratio_error(eigs, REFERENCE_SPECTRA[topology]['ratios']))
        except Exception:
            log_err = float('inf')

    expected = EXPECTED_BETTI.get(topology, None)
    k_filter = FILTER_K_BY_TOPOLOGY[topology]
    lap_filter = laplacian_eigenvector_filter(
        X, k=k_filter, eigenvectors=spec['eigenvectors'],
    )
    sweep_lap = mapper_sweep(
        X, lap_filter, topology=topology,
        n_intervals_grid=inner_n_intervals_grid, overlap_grid=overlap_grid,
        n_jobs=1,  # never spawn nested loky pools from a worker
    )
    cr_lap = correct_region(sweep_lap, expected) if expected else {
        'region_size': 0, 'region_fraction': 0.0,
        'n_correct_total': 0, 'total_configs': len(sweep_lap),
    }
    modal_lap = stable_betti(sweep_lap)

    return {
        'knn_k': int(knn_k),
        'sigma_factor': float(sigma_factor),
        'sigma_used': float(spec['sigma_used']),
        'spectral_log_ratio_error': log_err,
        'spectral_eigenvalues': [float(v) for v in eigs],
        'mapper_laplacian_correct_region': cr_lap,
        'mapper_laplacian_modal_betti': (int(modal_lap[0]), int(modal_lap[1])),
        'eigenvectors': spec['eigenvectors'],
    }


def _stage0_workerA(
    topology: str, knn_k: int, sigma_factor_grid: Sequence[float],
    n_samples: int, ambient_d: int, sigma_noise: float, seed: int,
    inner_n_intervals_grid: Sequence[int], overlap_grid: Sequence[float],
    spectral_K: int,
) -> tuple[str, int, list[dict]]:
    """One Stage A job: all sigma_factors for one (topology, knn_k).

    Pins BLAS to 1 thread so parallel and serial paths produce identical
    numerical results. kNN graph is computed once and shared across the
    sigma_factor variants (sigma only changes Gaussian weights, not the
    underlying graph).
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X, _gt = _sample_X_gt(topology, n_samples, ambient_d, sigma_noise, seed)
        knn_cache = compute_knn(X, knn_k)
        results: list[dict] = []
        for sf in sigma_factor_grid:
            spec = coifman_lafon_spectrum(
                X, knn_k=knn_k, K=spectral_K, sigma_factor=float(sf),
                precomputed_knn=knn_cache,
            )
            results.append(
                _score_inner(
                    X, topology, knn_k, float(sf), spec,
                    inner_n_intervals_grid, overlap_grid,
                )
            )
    return topology, int(knn_k), results


# ---------------------------------------------------------------------------
# Full-grid per-filter Mapper sweep (Stage B)
# ---------------------------------------------------------------------------

def _stage0_workerB(
    topology: str, filter_kind: str,
    n_samples: int, ambient_d: int, sigma_noise: float, seed: int,
    eigenvectors: np.ndarray | None,
    n_intervals_grid: Sequence[int], overlap_grid: Sequence[float],
    k_filter: int,
) -> tuple[str, str, dict]:
    """One Stage B job: full Mapper sweep at the winner's spectrum, for one
    filter (lap, gt, or pca).

    Re-derives X (and gt for the GT filter) from `seed` so we don't need to
    ship a (potentially large) X array through joblib pickle. The
    `eigenvectors` argument is only required for filter_kind='lap'; pass
    None for the other filters.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X, gt = _sample_X_gt(topology, n_samples, ambient_d, sigma_noise, seed)
        if filter_kind == 'lap':
            if eigenvectors is None:
                raise ValueError(
                    "filter_kind='lap' requires eigenvectors of the winner spectrum."
                )
            lens = laplacian_eigenvector_filter(X, k=k_filter, eigenvectors=eigenvectors)
        elif filter_kind == 'gt':
            if gt is None:
                raise ValueError(
                    f"GT filter requested for topology={topology!r} but the "
                    f"DGP returned no ground-truth coordinates."
                )
            lens = ground_truth_filter(gt)
        elif filter_kind == 'pca':
            lens = pca_filter(X, k=k_filter)
        else:
            raise ValueError(f"unknown filter_kind {filter_kind!r}")

        expected = EXPECTED_BETTI.get(topology, None)
        sweep = mapper_sweep(
            X, lens, topology=topology,
            n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
            n_jobs=1,  # never spawn nested loky pools from a worker
        )
        cr = correct_region(sweep, expected) if expected else None
        modal = stable_betti(sweep)
    return topology, filter_kind, {
        'correct_region': cr,
        'modal_betti': (int(modal[0]), int(modal[1])),
        'sweep': sweep,
    }


# ---------------------------------------------------------------------------
# Stage B FLAT — one job per (topology, filter, n_intervals, overlap).
# Used to push parallelism from 9 outer jobs (each running 24 Mapper configs
# sequentially) to 9 * 24 = 216 jobs across one loky pool. At n_jobs=56 the
# bottleneck Mapper sweep compresses ~6x, dominating the wall-clock budget.
# ---------------------------------------------------------------------------

def _stage0_flat_worker(
    topology: str, filter_kind: str, ni: int, ov: float,
    n_samples: int, ambient_d: int, sigma_noise: float, seed: int,
    lens: np.ndarray, distance_threshold: float, k_filter: int,
    eigenvectors: np.ndarray | None,
) -> tuple[str, str, int, float, dict]:
    """One flat Stage B job: run Mapper once at a single (ni, ov).

    The lens is precomputed once per (topology, filter) by the orchestrator
    and passed in. We re-derive X here from `seed` (cheap) so we don't
    pickle a (potentially large) X array per job. Everything stays bit-
    deterministic since the seed pins the sample.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X, _gt = _sample_X_gt(topology, n_samples, ambient_d, sigma_noise, seed)
        # `lens` is the precomputed filter values. For 'lap' it was computed
        # from `eigenvectors` at the winner; for 'gt' / 'pca' from X / gt
        # directly. We pass `eigenvectors` only for diagnostic; the lens is
        # the load-bearing input here.
        del eigenvectors
        key, entry = _mapper_one_config(
            X, lens, int(ni), float(ov),
            float(distance_threshold), topology,
        )
    return topology, filter_kind, key[0], key[1], entry


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def _dispatch(worker_fn, jobs: Sequence[tuple], n_jobs: int) -> list:
    """Run `worker_fn(*job)` for each job in `jobs`. Serial loop if n_jobs==1
    or len(jobs)<=1; otherwise joblib loky pool with n_jobs workers.

    The result list preserves input order regardless of dispatch path so
    downstream consumers see deterministic iteration.
    """
    if n_jobs == 1 or len(jobs) <= 1:
        return [worker_fn(*job) for job in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(worker_fn)(*job) for job in jobs
    )


# ---------------------------------------------------------------------------
# Winner pick + result assembly
# ---------------------------------------------------------------------------

def _pick_winner(entries: list[dict]) -> dict:
    """Among entries for one topology, pick the one with the largest
    correct-region fraction; tie-break with the smallest spectral log-ratio
    error. Replicates the original `_score_config`-driven tie-break.
    """
    def key(e: dict):
        cr = e['mapper_laplacian_correct_region']['region_fraction']
        err = e['spectral_log_ratio_error']
        err_for_rank = (
            err if (err is not None and np.isfinite(err)) else float('inf')
        )
        return (cr, -err_for_rank)
    return max(entries, key=key)


def _assemble_stage0_result(
    topology: str, n_samples: int, ambient_d: int, sigma_noise: float,
    winner: dict, tuning_log: list[dict],
    stage_b_for_topo: dict[str, dict],
    log_ratio_threshold: float, correct_region_threshold: float,
    multiplicity_eps: float = 0.05,
    multiplicity_n_clusters: int = 4,
) -> Stage0Result:
    from sae_topology.spectral import (
        first_nonzero_eigenvalue,
        multiplicity_clusters_match,
        near_zero_count as near_zero_count_fn,
    )
    from sae_topology.mapper import failure_mode_diagnostic

    cr_lap = (
        stage_b_for_topo['lap']['correct_region']
        or winner['mapper_laplacian_correct_region']
    )
    cr_frac = cr_lap['region_fraction']
    is_interior = bool(cr_lap.get('is_interior', False))
    log_err = winner['spectral_log_ratio_error']
    has_ref = topology in REFERENCE_SPECTRA

    spectral_pass = (
        True if not has_ref
        else (log_err is not None and log_err < log_ratio_threshold)
    )
    mapper_pass = bool(cr_frac >= correct_region_threshold)

    cr_gt = (
        stage_b_for_topo['gt']['correct_region']
        if 'gt' in stage_b_for_topo else None
    )
    cr_pca = stage_b_for_topo['pca']['correct_region']
    modal_lap = stage_b_for_topo['lap']['modal_betti']
    sweep_lap = stage_b_for_topo['lap'].get('sweep')

    # ----- New validation criteria (stage0_tuning.md §4.2 + §4.1) ----------
    eigs = np.asarray(winner['spectral_eigenvalues'], dtype=float)

    if has_ref:
        try:
            mult_check = multiplicity_clusters_match(
                eigs,
                REFERENCE_SPECTRA[topology]['levels'],
                REFERENCE_SPECTRA[topology]['mults'],
                eps=multiplicity_eps,
                n_clusters=multiplicity_n_clusters,
            )
            multiplicity_pass = bool(mult_check['all_match'])
        except Exception as e:
            mult_check = {'error': str(e), 'all_match': False, 'per_cluster': []}
            multiplicity_pass = False
    else:
        # No closed-form reference (e.g., figure_eight): not enforced.
        mult_check = None
        multiplicity_pass = True

    # Near-zero count: empirical eigenvalues below 0.1 * lambda_1 must equal
    # b_0 of the manifold (1 for all single-component manifolds in Stage 0).
    expected_b0 = (EXPECTED_BETTI.get(topology, (1, 0))[0])
    lam1 = first_nonzero_eigenvalue(eigs, zero_threshold=1e-8)
    if lam1 is not None and lam1 > 0:
        nz = near_zero_count_fn(eigs, threshold=0.1 * lam1)
    else:
        nz = int((eigs < 1e-8).sum())
    near_zero_pass = bool(nz == expected_b0)

    expected_betti = EXPECTED_BETTI.get(topology, None)
    if sweep_lap and expected_betti:
        failure_diag = failure_mode_diagnostic(sweep_lap, expected_betti)
    else:
        failure_diag = None

    mapper_interior_pass = bool(is_interior)

    overall_pass = bool(
        spectral_pass and mapper_pass and mapper_interior_pass
        and multiplicity_pass and near_zero_pass
    )

    return Stage0Result(
        topology=topology,
        n_samples=n_samples,
        ambient_d=ambient_d,
        sigma_noise=sigma_noise,
        knn_k=winner['knn_k'],
        sigma_factor=winner['sigma_factor'],
        sigma_used=winner['sigma_used'],
        spectral_log_ratio_error=log_err,
        spectral_eigenvalues=winner['spectral_eigenvalues'],
        mapper_laplacian_correct_region=cr_lap,
        mapper_gt_correct_region=cr_gt,
        mapper_pca_correct_region=cr_pca,
        mapper_laplacian_modal_betti=modal_lap,
        spectral_pass=spectral_pass,
        mapper_pass=mapper_pass,
        overall_pass=overall_pass,
        tuning_log=tuning_log,
        multiplicity_pass=multiplicity_pass,
        multiplicity_check=mult_check,
        near_zero_pass=near_zero_pass,
        near_zero_count=int(nz),
        mapper_interior_pass=mapper_interior_pass,
        failure_mode_diagnostic=failure_diag,
    )


# ---------------------------------------------------------------------------
# Single-topology synchronous wrapper (back-compat)
# ---------------------------------------------------------------------------

def stage0_for_topology(
    topology: str,
    n_samples: int | None = None,
    ambient_d: int = 64,
    sigma: float = 0.01,
    seed: int = 0,
    knn_k_grid: Sequence[int] = DEFAULT_KNN_K_GRID,
    sigma_factor_grid: Sequence[float] = DEFAULT_SIGMA_FACTOR_GRID,
    n_intervals_grid: Sequence[int] = N_INTERVALS_GRID,
    overlap_grid: Sequence[float] = OVERLAP_GRID,
    inner_n_intervals_grid: Sequence[int] = INNER_NI_GRID,
    spectral_K: int = 25,
    log_ratio_threshold: float = 0.05,
    correct_region_threshold: float = 0.50,
    multiplicity_eps: float = 0.05,
    multiplicity_n_clusters: int = 4,
) -> Stage0Result:
    """Run Stage 0 auto-tune for one topology, single-process.

    Equivalent to `run_stage0([topology], n_jobs=1, ...)[topology]` but with
    a more focused per-topology API for callers that want one-off
    diagnostics (e.g. notebooks).
    """
    return run_stage0(
        topologies=(topology,),
        ambient_d=ambient_d, sigma=sigma, seed=seed,
        knn_k_grid=knn_k_grid, sigma_factor_grid=sigma_factor_grid,
        n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
        inner_n_intervals_grid=inner_n_intervals_grid,
        spectral_K=spectral_K,
        log_ratio_threshold=log_ratio_threshold,
        correct_region_threshold=correct_region_threshold,
        multiplicity_eps=multiplicity_eps,
        multiplicity_n_clusters=multiplicity_n_clusters,
        n_samples_override={topology: n_samples} if n_samples is not None else None,
        n_jobs=1,
    )[topology]


# ---------------------------------------------------------------------------
# Multi-topology orchestrator
# ---------------------------------------------------------------------------

def run_stage0(
    topologies: Sequence[str] = DEFAULT_STAGE0_TOPOLOGIES,
    out_yaml: Path | None = None,
    out_json: Path | None = None,
    quick: bool = False,
    n_jobs: int = 1,
    ambient_d: int = 64,
    sigma: float = 0.01,
    seed: int = 0,
    knn_k_grid: Sequence[int] = DEFAULT_KNN_K_GRID,
    sigma_factor_grid: Sequence[float] = DEFAULT_SIGMA_FACTOR_GRID,
    n_intervals_grid: Sequence[int] = N_INTERVALS_GRID,
    overlap_grid: Sequence[float] = OVERLAP_GRID,
    inner_n_intervals_grid: Sequence[int] = INNER_NI_GRID,
    spectral_K: int = 25,
    log_ratio_threshold: float = 0.05,
    correct_region_threshold: float = 0.50,
    multiplicity_eps: float = 0.05,
    multiplicity_n_clusters: int = 4,
    n_samples_override: dict[str, int] | None = None,
    capture_sweeps_into: dict | None = None,
    capture_winners_into: dict | None = None,
) -> dict[str, Stage0Result]:
    """Two-stage flat-parallel Stage 0.

    Stage A: 4 x 3 = 12 jobs scoring (knn_k, sigma_factor) per topology
             (sigma_factor variants share the kNN inside one job).
    Stage B: 4 x 3 - 1 = 11 jobs running the full Mapper sweep on each filter
             at the winning (knn_k, sigma_factor) per topology.

    `n_jobs=1` is a serial loop over the same workers (identical results).
    """
    # Resolve per-topology N (with `--quick` halving and per-call override).
    n_samples_by_topo: dict[str, int] = {}
    for topo in topologies:
        if n_samples_override and n_samples_override.get(topo) is not None:
            n_samples_by_topo[topo] = int(n_samples_override[topo])
        else:
            n = DEFAULT_SAMPLE_SIZES.get(topo, 2_000)
            if quick:
                n = max(n // 2, 1_000)
            n_samples_by_topo[topo] = n

    # ------- Stage A: build flat job list ----------------------------------
    knn_k_grid_t = tuple(int(k) for k in knn_k_grid)
    sigma_factor_grid_t = tuple(float(s) for s in sigma_factor_grid)
    inner_ni_grid_t = tuple(int(n) for n in inner_n_intervals_grid)
    overlap_grid_t = tuple(float(o) for o in overlap_grid)
    n_intervals_grid_t = tuple(int(n) for n in n_intervals_grid)

    stage_a_jobs = [
        (
            topo, kk, sigma_factor_grid_t,
            n_samples_by_topo[topo], ambient_d, sigma, seed,
            inner_ni_grid_t, overlap_grid_t, spectral_K,
        )
        for topo in topologies
        for kk in knn_k_grid_t
    ]
    print(f"\n=== Stage 0: dispatching {len(stage_a_jobs)} Stage A jobs "
          f"(n_jobs={n_jobs}) ===", flush=True)
    stage_a_outputs = _dispatch(_stage0_workerA, stage_a_jobs, n_jobs)

    # ------- Group + winner pick (serial) ----------------------------------
    entries_by_topo: dict[str, list[dict]] = {topo: [] for topo in topologies}
    for topo, _kk, entries in stage_a_outputs:
        entries_by_topo[topo].extend(entries)

    winners: dict[str, dict] = {}
    tuning_logs: dict[str, list[dict]] = {}
    for topo in topologies:
        # Print per-topology auto-tune log in the same format as before.
        print(f"\n[{topo}] auto-tune scored {len(entries_by_topo[topo])} configs:",
              flush=True)
        log_no_ev: list[dict] = []
        for entry in entries_by_topo[topo]:
            cr_frac = entry['mapper_laplacian_correct_region']['region_fraction']
            err = entry['spectral_log_ratio_error']
            err_for_rank = (
                err if (err is not None and np.isfinite(err)) else float('inf')
            )
            print(f"    knn_k={entry['knn_k']}, sigma_factor={entry['sigma_factor']}: "
                  f"Lap correct-region={cr_frac*100:.0f}% "
                  f"({len(inner_ni_grid_t) * len(overlap_grid_t)} configs); "
                  f"spectral E={err_for_rank:.4f}", flush=True)
            log_no_ev.append({k: v for k, v in entry.items() if k != 'eigenvectors'})
        tuning_logs[topo] = log_no_ev
        winners[topo] = _pick_winner(entries_by_topo[topo])
        print(f"  [{topo}] winner: knn_k={winners[topo]['knn_k']}, "
              f"sigma_factor={winners[topo]['sigma_factor']}", flush=True)

    # ------- Stage B prep (serial, cheap): precompute lens + threshold -----
    # We compute the lens (filter values) once per (topology, filter) here
    # so that the flat 216-job pool below only re-derives X (cheap from
    # seed) inside each worker. The lens is the load-bearing input; passing
    # it via joblib pickle is small (max ~3 MB per (topology, filter)).
    print(f"\n=== Stage 0: pre-computing lens + threshold per (topology, filter) "
          f"(serial) ===", flush=True)
    lens_by_pair: dict[tuple[str, str], np.ndarray] = {}
    threshold_by_topo: dict[str, float] = {}
    k_filter_by_topo: dict[str, int] = {}
    for topo in topologies:
        winner = winners[topo]
        k_filter = FILTER_K_BY_TOPOLOGY[topo]
        k_filter_by_topo[topo] = k_filter
        X_topo, gt_topo = _sample_X_gt(
            topo, n_samples_by_topo[topo], ambient_d, sigma, seed,
        )
        threshold_by_topo[topo] = float(global_distance_threshold(X_topo))
        # Laplacian filter: derive from the winner's eigenvectors.
        lens_by_pair[(topo, 'lap')] = laplacian_eigenvector_filter(
            X_topo, k=k_filter, eigenvectors=winner['eigenvectors'],
        )
        # GT filter: only for manifolds that surface intrinsic coords.
        if topo in GT_FILTER_TOPOLOGIES and gt_topo is not None:
            lens_by_pair[(topo, 'gt')] = ground_truth_filter(gt_topo)
        # PCA filter: top-k principal components of X.
        lens_by_pair[(topo, 'pca')] = pca_filter(X_topo, k=k_filter)

    # ------- Stage B FLAT: 216 jobs in one loky pool -----------------------
    flat_jobs: list[tuple] = []
    for (topo, fk), lens in lens_by_pair.items():
        for ni in n_intervals_grid_t:
            for ov in overlap_grid_t:
                flat_jobs.append((
                    topo, fk, int(ni), float(ov),
                    n_samples_by_topo[topo], ambient_d, sigma, seed,
                    lens, threshold_by_topo[topo], k_filter_by_topo[topo],
                    None,  # eigenvectors arg, unused once lens is precomputed
                ))
    print(f"\n=== Stage 0: dispatching {len(flat_jobs)} flat Stage B jobs "
          f"(n_jobs={n_jobs}) ===", flush=True)
    flat_outputs = _dispatch(_stage0_flat_worker, flat_jobs, n_jobs)

    # ------- Reassemble per-(topology, filter) sweep dicts ----------------
    sweep_by_pair: dict[tuple[str, str], dict] = {pair: {} for pair in lens_by_pair}
    for topo, fk, ni, ov, entry in flat_outputs:
        sweep_by_pair[(topo, fk)][(int(ni), float(ov))] = entry

    stage_b_by_topo: dict[str, dict[str, dict]] = {topo: {} for topo in topologies}
    for (topo, fk), sweep in sweep_by_pair.items():
        expected = EXPECTED_BETTI.get(topo, None)
        cr = correct_region(sweep, expected) if expected else None
        modal = stable_betti(sweep)
        stage_b_by_topo[topo][fk] = {
            'correct_region': cr,
            'modal_betti': (int(modal[0]), int(modal[1])),
            'sweep': sweep,
        }

    # Side-channel: let the orchestrator capture the raw sweeps + winners
    # so it can re-use them for plotting / artifact persistence without
    # re-running Stage B. Backwards compatible: existing callers pass None.
    if capture_sweeps_into is not None:
        for (topo, fk), sweep in sweep_by_pair.items():
            capture_sweeps_into.setdefault(topo, {})[fk] = sweep
    if capture_winners_into is not None:
        for topo, w in winners.items():
            # eigenvectors are heavy; expose them so the orchestrator skips
            # one eigsh call per topology during artifact persistence.
            capture_winners_into[topo] = {
                'knn_k': int(w['knn_k']),
                'sigma_factor': float(w['sigma_factor']),
                'sigma_used': float(w['sigma_used']),
                'spectral_eigenvalues': list(w['spectral_eigenvalues']),
                'eigenvectors': w['eigenvectors'],
            }

    # ------- Assemble per-topology Stage0Result + report -------------------
    results: dict[str, Stage0Result] = {}
    for topo in topologies:
        winner = winners[topo]
        res = _assemble_stage0_result(
            topology=topo,
            n_samples=n_samples_by_topo[topo],
            ambient_d=ambient_d,
            sigma_noise=sigma,
            winner=winner,
            tuning_log=tuning_logs[topo],
            stage_b_for_topo=stage_b_by_topo[topo],
            log_ratio_threshold=log_ratio_threshold,
            correct_region_threshold=correct_region_threshold,
            multiplicity_eps=multiplicity_eps,
            multiplicity_n_clusters=multiplicity_n_clusters,
        )
        results[topo] = res
        cr = res.mapper_laplacian_correct_region
        gt_str = ''
        if res.mapper_gt_correct_region is not None:
            gt_str = (f"  GT correct-region: "
                      f"{res.mapper_gt_correct_region['region_fraction']*100:.0f}% "
                      f"(size={res.mapper_gt_correct_region['region_size']})")
        print(f"\n[{topo}] (N={res.n_samples})")
        print(f"  best: knn_k={res.knn_k}, sigma_factor={res.sigma_factor}")
        if res.spectral_log_ratio_error is not None:
            print(f"  spectral E={res.spectral_log_ratio_error:.4f} -> "
                  f"{'PASS' if res.spectral_pass else 'FAIL'}")
        else:
            print(f"  spectral: no closed-form reference (Mapper-only)")
        print(f"  mapper Laplacian correct-region: {cr['region_fraction']*100:.0f}% "
              f"(size={cr['region_size']}/{cr['total_configs']}, "
              f"n_correct_anywhere={cr['n_correct_total']})  -> "
              f"{'PASS' if res.mapper_pass else 'FAIL'}  "
              f"interior={'PASS' if res.mapper_interior_pass else 'FAIL'}")
        print(f"  multiplicity (first {multiplicity_n_clusters} clusters): "
              f"{'PASS' if res.multiplicity_pass else 'FAIL'}  "
              f"near-zero count={res.near_zero_count} "
              f"({'PASS' if res.near_zero_pass else 'FAIL'})")
        print(f"  modal Betti (Laplacian): {res.mapper_laplacian_modal_betti}{gt_str}")
        print(f"  overall: {'PASS' if res.overall_pass else 'FAIL'}")

    if out_yaml is not None:
        out_yaml = Path(out_yaml)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            'n_samples': max(r.n_samples for r in results.values()) if results else None,
            'ambient_d': max(r.ambient_d for r in results.values()) if results else None,
            'n_intervals_grid': list(n_intervals_grid_t),
            'overlap_grid': list(overlap_grid_t),
            'knn_k_grid': list(knn_k_grid_t),
            'sigma_factor_grid': list(sigma_factor_grid_t),
            'spectral_K': spectral_K,
            'multiplicity_eps': multiplicity_eps,
            'multiplicity_n_clusters': multiplicity_n_clusters,
            'sigma_rule': 'median_heuristic',
            'laplacian': 'coifman_lafon_alpha1',
            'cluster_rule': 'single_linkage_global_5x_median_knn',
        }
        baseline = {'_meta': meta}
        baseline.update({
            topo: {
                'n_samples': r.n_samples,
                'ambient_d': r.ambient_d,
                'sigma_noise': r.sigma_noise,
                'knn_k': r.knn_k,
                'sigma_factor': r.sigma_factor,
                'sigma_used': r.sigma_used,
                'correct_region_fraction': r.mapper_laplacian_correct_region['region_fraction'],
                'correct_region_size': r.mapper_laplacian_correct_region['region_size'],
                'mapper_interior_pass': r.mapper_interior_pass,
                'spectral_log_ratio_error': r.spectral_log_ratio_error,
                'mapper_laplacian_modal_betti': list(r.mapper_laplacian_modal_betti),
                'multiplicity_pass': r.multiplicity_pass,
                'near_zero_pass': r.near_zero_pass,
                'near_zero_count': r.near_zero_count,
                'spectral_pass': r.spectral_pass,
                'mapper_pass': r.mapper_pass,
                'overall_pass': r.overall_pass,
            }
            for topo, r in results.items()
        })
        with open(out_yaml, 'w') as f:
            yaml.safe_dump(baseline, f, sort_keys=False)
        print(f"\nWrote baseline to {out_yaml}")

    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(
                {topo: asdict(r) for topo, r in results.items()},
                f, indent=2, default=str,
            )
        print(f"Wrote full Stage-0 detail to {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 0 pipeline validation (auto-tune)")
    parser.add_argument('--topologies', nargs='+',
                        default=list(DEFAULT_STAGE0_TOPOLOGIES))
    parser.add_argument('--ambient_d', type=int, default=64)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--out_yaml', default='configs/stage0/baseline.yaml')
    parser.add_argument('--out_json', default='configs/stage0/baseline.json')
    parser.add_argument('--quick', action='store_true',
                        help='Halve sample sizes for development.')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Worker count for the joblib loky pool. '
                             '1 = serial. Set to a value <= number of CPUs.')
    args = parser.parse_args()

    run_stage0(
        topologies=args.topologies,
        out_yaml=Path(args.out_yaml),
        out_json=Path(args.out_json),
        ambient_d=args.ambient_d,
        sigma=args.sigma,
        quick=args.quick,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()
