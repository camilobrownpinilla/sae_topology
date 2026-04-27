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

Run:  python -m sae_topology.experiments.stage0_validate
      python -m sae_topology.experiments.stage0_validate --quick   (N/2 for dev)

Writes:  configs/stage0/baseline.yaml
         configs/stage0/baseline.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
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
from sae_topology.spectral import (
    coifman_lafon_spectrum,
    log_ratio_error,
    multiplicity_check,
    near_zero_count,
    REFERENCE_SPECTRA,
    GROUND_TRUTH_BETTI,
)


DEFAULT_SAMPLE_SIZES = {
    'circle': 2_000,
    'sphere': 10_000,
    'torus':  10_000,
    'figure_eight': 2_000,
}

# Auto-tune grid (per spec section 7.1 plus our extension).
DEFAULT_KNN_K_GRID = (15, 25, 40)
DEFAULT_SIGMA_FACTOR_GRID = (0.5, 1.0, 2.0)

FILTER_K_BY_TOPOLOGY = {
    'circle': 3,
    'torus':  4,
    'sphere': 4,
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


def _dgp_kwargs(topology: str) -> dict:
    if topology == 'torus':
        return {'major_radius': 1.0, 'minor_radius': 1.0}
    if topology == 'sphere':
        return {'radius': 1.0}
    return {}


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


def _score_config(
    X: np.ndarray, topology: str,
    knn_k: int, sigma_factor: float,
    n_intervals_grid, overlap_grid, spectral_K: int,
) -> dict:
    """Cheap scoring pass for the auto-tune inner loop. Computes ONLY:
      - spectral log-ratio error
      - Mapper-Laplacian correct-region fraction at a *reduced* (ni, ov) grid

    Skips GT and PCA filter sweeps; those run once at the winning
    (knn_k, sigma_factor) inside `_full_diagnostics`.
    """
    spec = coifman_lafon_spectrum(
        X, knn_k=knn_k, K=spectral_K, sigma_factor=sigma_factor,
    )
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
        n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
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
        'eigenvectors': spec['eigenvectors'],  # passed through for full-grid reuse
    }


def _full_diagnostics(
    X: np.ndarray, gt: np.ndarray | None, topology: str,
    best: dict,
    n_intervals_grid, overlap_grid,
) -> dict:
    """At the auto-tune winner, run the FULL Mapper sweep on Laplacian
    + GT + PCA filters and return the augmented diagnostics."""
    expected = EXPECTED_BETTI.get(topology, None)
    k_filter = FILTER_K_BY_TOPOLOGY[topology]

    lap_filter = laplacian_eigenvector_filter(
        X, k=k_filter, eigenvectors=best['eigenvectors'],
    )
    sweep_lap = mapper_sweep(
        X, lap_filter, topology=topology,
        n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
    )
    cr_lap = correct_region(sweep_lap, expected) if expected else None
    modal_lap = stable_betti(sweep_lap)

    cr_gt = None
    if gt is not None and topology in {'circle', 'torus', 'sphere'}:
        gt_lens = ground_truth_filter(gt)
        sweep_gt = mapper_sweep(
            X, gt_lens, topology=topology,
            n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
        )
        cr_gt = correct_region(sweep_gt, expected) if expected else None

    pca_lens = pca_filter(X, k=k_filter)
    sweep_pca = mapper_sweep(
        X, pca_lens, topology=topology,
        n_intervals_grid=n_intervals_grid, overlap_grid=overlap_grid,
    )
    cr_pca = correct_region(sweep_pca, expected) if expected else None

    return {
        'mapper_laplacian_correct_region': cr_lap,
        'mapper_laplacian_modal_betti': (int(modal_lap[0]), int(modal_lap[1])),
        'mapper_gt_correct_region': cr_gt,
        'mapper_pca_correct_region': cr_pca,
    }


# Inner-loop reduced grids (used for cheap scoring during auto-tune).
INNER_NI_GRID = (5, 8, 12)


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
    spectral_K: int = 20,
    log_ratio_threshold: float = 0.05,
    correct_region_threshold: float = 0.50,
) -> Stage0Result:
    """Run Stage 0 auto-tune for one topology.

    Two-stage search:
      1. Inner loop over (knn_k, sigma_factor): cheap scoring on a reduced
         (ni, ov) grid (default ni in {5, 8, 12}) using only the Laplacian
         filter. ~5x faster than scoring on the full grid + all 3 filters.
      2. At the winning (knn_k, sigma_factor): re-run the full 20-config
         (n_intervals, overlap) grid on Laplacian + GT + PCA for the
         headline diagnostics.

    The winner is picked by max correct-region fraction (Laplacian filter,
    inner grid); ties broken by lowest spectral log-ratio error.
    """
    if n_samples is None:
        n_samples = DEFAULT_SAMPLE_SIZES.get(topology, 2_000)

    np.random.seed(seed)
    dgp = make_dgp(
        topology, d=ambient_d, sigma=sigma, c0=np.zeros(ambient_d),
        **_dgp_kwargs(topology),
    )
    X, gt = dgp.sample_with_gt(n_samples)
    has_ref = topology in REFERENCE_SPECTRA

    tuning_log: list = []
    best = None
    n_total = len(list(knn_k_grid)) * len(list(sigma_factor_grid))
    seen = 0
    for kk in knn_k_grid:
        for sf in sigma_factor_grid:
            seen += 1
            print(f"  [{topology}] auto-tune {seen}/{n_total}: "
                  f"knn_k={kk}, sigma_factor={sf}", flush=True)
            entry = _score_config(
                X, topology, int(kk), float(sf),
                inner_n_intervals_grid, overlap_grid, spectral_K,
            )
            log_entry = {k: v for k, v in entry.items() if k != 'eigenvectors'}
            tuning_log.append(log_entry)
            cr_frac = entry['mapper_laplacian_correct_region']['region_fraction']
            err = entry['spectral_log_ratio_error']
            err_for_rank = err if (err is not None and np.isfinite(err)) else float('inf')
            print(f"    -> Lap correct-region={cr_frac*100:.0f}% "
                  f"(reduced grid, {len(inner_n_intervals_grid) * len(overlap_grid)} configs); "
                  f"spectral E={err_for_rank:.4f}", flush=True)
            score = (cr_frac, -err_for_rank)
            if best is None or score > (
                best['mapper_laplacian_correct_region']['region_fraction'],
                -(best['spectral_log_ratio_error']
                  if (best['spectral_log_ratio_error'] is not None
                      and np.isfinite(best['spectral_log_ratio_error']))
                  else float('inf')),
            ):
                best = entry

    print(f"  [{topology}] winner: knn_k={best['knn_k']}, "
          f"sigma_factor={best['sigma_factor']}; running full 20-config grid + GT + PCA",
          flush=True)
    full = _full_diagnostics(
        X, gt, topology, best,
        n_intervals_grid, overlap_grid,
    )

    cr_lap = full['mapper_laplacian_correct_region'] or best['mapper_laplacian_correct_region']
    cr_frac = cr_lap['region_fraction']
    log_err = best['spectral_log_ratio_error']

    spectral_pass = (
        True if not has_ref
        else (log_err is not None and log_err < log_ratio_threshold)
    )
    mapper_pass = bool(cr_frac >= correct_region_threshold)

    return Stage0Result(
        topology=topology,
        n_samples=n_samples,
        ambient_d=ambient_d,
        sigma_noise=sigma,
        knn_k=best['knn_k'],
        sigma_factor=best['sigma_factor'],
        sigma_used=best['sigma_used'],
        spectral_log_ratio_error=log_err,
        spectral_eigenvalues=best['spectral_eigenvalues'],
        mapper_laplacian_correct_region=cr_lap,
        mapper_gt_correct_region=full['mapper_gt_correct_region'],
        mapper_pca_correct_region=full['mapper_pca_correct_region'],
        mapper_laplacian_modal_betti=full['mapper_laplacian_modal_betti'],
        spectral_pass=spectral_pass,
        mapper_pass=mapper_pass,
        overall_pass=bool(spectral_pass and mapper_pass),
        tuning_log=tuning_log,
    )


def run_stage0(
    topologies: Sequence[str] = ('circle', 'torus', 'sphere', 'figure_eight'),
    out_yaml: Path | None = None,
    out_json: Path | None = None,
    quick: bool = False,
    **per_topology_kwargs,
) -> dict[str, Stage0Result]:
    results: dict[str, Stage0Result] = {}
    for topo in topologies:
        n = DEFAULT_SAMPLE_SIZES.get(topo, 2_000)
        if quick:
            n = max(n // 2, 1_000)
        print(f"\n=== Stage 0: {topo} (N={n}) ===", flush=True)
        res = stage0_for_topology(topo, n_samples=n, **per_topology_kwargs)
        results[topo] = res
        cr = res.mapper_laplacian_correct_region
        gt_str = ''
        if res.mapper_gt_correct_region is not None:
            gt_str = (f"  GT correct-region: {res.mapper_gt_correct_region['region_fraction']*100:.0f}% "
                      f"(size={res.mapper_gt_correct_region['region_size']})")
        print(f"  best: knn_k={res.knn_k}, sigma_factor={res.sigma_factor}")
        if res.spectral_log_ratio_error is not None:
            print(f"  spectral E={res.spectral_log_ratio_error:.4f} -> "
                  f"{'PASS' if res.spectral_pass else 'FAIL'}")
        else:
            print(f"  spectral: no closed-form reference (Mapper-only)")
        print(f"  mapper Laplacian correct-region: {cr['region_fraction']*100:.0f}% "
              f"(size={cr['region_size']}/{cr['total_configs']}, "
              f"n_correct_anywhere={cr['n_correct_total']})  -> "
              f"{'PASS' if res.mapper_pass else 'FAIL'}")
        print(f"  modal Betti (Laplacian): {res.mapper_laplacian_modal_betti}{gt_str}")
        print(f"  overall: {'PASS' if res.overall_pass else 'FAIL'}")

    if out_yaml is not None:
        out_yaml = Path(out_yaml)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        baseline = {
            topo: {
                'n_samples': r.n_samples,
                'ambient_d': r.ambient_d,
                'sigma_noise': r.sigma_noise,
                'knn_k': r.knn_k,
                'sigma_factor': r.sigma_factor,
                'sigma_used': r.sigma_used,
                'correct_region_fraction': r.mapper_laplacian_correct_region['region_fraction'],
                'correct_region_size': r.mapper_laplacian_correct_region['region_size'],
                'spectral_log_ratio_error': r.spectral_log_ratio_error,
                'mapper_laplacian_modal_betti': list(r.mapper_laplacian_modal_betti),
                'spectral_pass': r.spectral_pass,
                'mapper_pass': r.mapper_pass,
                'overall_pass': r.overall_pass,
            }
            for topo, r in results.items()
        }
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
                        default=['circle', 'torus', 'sphere', 'figure_eight'])
    parser.add_argument('--ambient_d', type=int, default=64)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--out_yaml', default='configs/stage0/baseline.yaml')
    parser.add_argument('--out_json', default='configs/stage0/baseline.json')
    parser.add_argument('--quick', action='store_true',
                        help='Halve sample sizes for development.')
    args = parser.parse_args()

    run_stage0(
        topologies=args.topologies,
        out_yaml=Path(args.out_yaml),
        out_json=Path(args.out_json),
        ambient_d=args.ambient_d,
        sigma=args.sigma,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()
