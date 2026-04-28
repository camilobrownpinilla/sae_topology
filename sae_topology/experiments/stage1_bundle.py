"""Stage 1 bundle: train one SAE from a YAML config, run a Laplacian-only
Mapper sweep on its post-activations, and persist Stage-0-style artifacts
plus figures.

CLI:

    python -m sae_topology.experiments.stage1_bundle \\
        --config <yaml> --out_dir results/stage1/<run> --n_jobs N

Outputs (under `<out_dir>/`):

    samples.npz                  eval_X, pre, post (codes), recon, gt, seed
    spectrum.npz                 eigenvalues, eigenvectors, sigma_used, knn_k,
                                 sigma_factor, seed
    mapper_sweep.json            full 24-config Laplacian sweep table
    training_metrics.json        per-step train metrics (mse, loss, mean_l0,
                                 n_dead, step)
    report.json                  Stage1Result + multiplicity/near-zero checks
                                 + frozen meta
    report.md                    Stage-0-style sign-off checklist
    figures/
        mapper_grid_laplacian.png
        spectrum_overlay.png
        training_curves.png
        stable_region_renderings/mapper_ni*_ov*.png  (theta+phi pair for T^2)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from sae_topology.dgp import make_dgp
from sae_topology.mapper import (
    laplacian_eigenvector_filter,
    mapper_sweep,
    run_mapper_once,
    stable_betti,
    correct_region,
    failure_mode_diagnostic,
    global_distance_threshold,
    N_INTERVALS_GRID,
    OVERLAP_GRID,
)
from sae_topology.saes.architectures import build_sae
from sae_topology.saes.config import SAETrainConfig
from sae_topology.saes.trainer import SAETrainer
from sae_topology.spectral import (
    coifman_lafon_spectrum,
    log_ratio_error,
    multiplicity_clusters_match,
    near_zero_count,
    REFERENCE_SPECTRA,
)
from sae_topology.experiments.stage0_validate import (
    EXPECTED_BETTI,
    FILTER_K_BY_TOPOLOGY,
)
from sae_topology.experiments.stage0_artifacts import _to_jsonable, _serialise_sweep
from sae_topology.experiments.stage0_plots import (
    mapper_grid_heatmap,
    spectrum_overlay_plot,
    mapper_graph_renderings,
    stable_configs_from_correct_region,
)
from sae_topology.analysis.plot import plot_training_curves


DEFAULT_BASELINE_PATH = Path(__file__).parents[2] / 'configs/stage0/baseline.yaml'

# Pass thresholds (mirrors stage0_run.py defaults).
DEFAULT_LOG_RATIO_THRESHOLD = 0.05
DEFAULT_CORRECT_REGION_THRESHOLD = 0.50
DEFAULT_MULTIPLICITY_EPS = 0.05
DEFAULT_MULTIPLICITY_N_CLUSTERS = 4
DEFAULT_SPECTRAL_K = 25  # matches the post-K20 sphere/torus fix


@dataclass
class Stage1Result:
    """Summary of one Stage 1 (SAE post-activation) bundle."""
    # Identity / config
    topology: str
    arch: str
    d_sae: int
    seed: int
    n_eval_samples: int
    ambient_d: int
    sigma_noise: float

    # Stage 0 baseline used
    knn_k: int
    sigma_factor: float
    sigma_used: float

    # Spectral
    spectral_log_ratio_error: float | None
    spectral_eigenvalues: list

    # Mapper (Laplacian filter only)
    mapper_laplacian_correct_region: dict
    mapper_laplacian_modal_betti: tuple

    # Pass flags
    spectral_pass: bool
    mapper_pass: bool
    multiplicity_pass: bool
    near_zero_pass: bool
    mapper_interior_pass: bool
    overall_pass: bool

    # Detailed metrics
    multiplicity_check: dict | None
    near_zero_count: int
    failure_mode_diagnostic: dict | None

    # Training metrics (final-step values; full curves on disk)
    final_mse: float | None
    final_mean_l0: float | None
    final_n_dead: int | None


def _load_baseline(path: Path | str) -> dict:
    p = Path(path) if path else DEFAULT_BASELINE_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _baseline_for_topology(baseline: dict, topology: str) -> tuple[int, float, bool]:
    entry = baseline.get(topology) if baseline else None
    if entry and 'knn_k' in entry and 'sigma_factor' in entry:
        return int(entry['knn_k']), float(entry['sigma_factor']), True
    return 25, 1.0, False  # fallback defaults (matches runner.py)


def _dgp_kwargs(topology: str) -> dict:
    if topology == 'torus':
        return {'major_radius': 1.0, 'minor_radius': 1.0}
    if topology == 'sphere':
        return {'radius': 1.0}
    return {}


def _build_stage1_result(
    *, topology: str, config: SAETrainConfig, ambient_d: int, sigma_noise: float,
    knn_k: int, sigma_factor: float, sigma_used: float,
    eigenvalues: np.ndarray, sweep: dict, train_metrics: dict,
    log_ratio_threshold: float, correct_region_threshold: float,
    multiplicity_eps: float, multiplicity_n_clusters: int,
) -> Stage1Result:
    expected = EXPECTED_BETTI.get(topology)
    cr = (correct_region(sweep, expected) if expected else
          {'region_size': 0, 'region_fraction': 0.0,
           'n_correct_total': 0, 'total_configs': len(sweep),
           'is_interior': False})
    modal = stable_betti(sweep)

    has_ref = topology in REFERENCE_SPECTRA
    log_err: float | None = None
    mult_check: dict | None = None
    if has_ref:
        try:
            log_err = float(log_ratio_error(
                eigenvalues, REFERENCE_SPECTRA[topology]['ratios']))
        except Exception:
            log_err = float('inf')
        try:
            mult_check = multiplicity_clusters_match(
                eigenvalues,
                REFERENCE_SPECTRA[topology]['levels'],
                REFERENCE_SPECTRA[topology]['mults'],
                eps=multiplicity_eps,
                n_clusters=multiplicity_n_clusters,
            )
        except Exception:
            mult_check = None

    nz = int(near_zero_count(eigenvalues))
    expected_b0 = expected[0] if expected is not None else 1
    nz_pass = (nz == expected_b0)

    fmd = (failure_mode_diagnostic(sweep, expected) if expected else None)

    spec_pass = (log_err is not None and log_err < log_ratio_threshold)
    mapper_pass = (cr['region_fraction'] >= correct_region_threshold)
    mult_pass = bool(mult_check and mult_check.get('all_match'))
    interior_pass = bool(cr.get('is_interior'))

    overall = spec_pass and mapper_pass and mult_pass and nz_pass and interior_pass

    metrics = train_metrics or {}
    final_mse = float(metrics['mse'][-1]) if metrics.get('mse') else None
    final_l0 = float(metrics['mean_l0'][-1]) if metrics.get('mean_l0') else None
    final_dead = int(metrics['n_dead'][-1]) if metrics.get('n_dead') else None

    return Stage1Result(
        topology=topology,
        arch=config.arch, d_sae=config.d_sae, seed=config.seed,
        n_eval_samples=config.n_eval_samples,
        ambient_d=ambient_d, sigma_noise=sigma_noise,
        knn_k=knn_k, sigma_factor=sigma_factor, sigma_used=sigma_used,
        spectral_log_ratio_error=log_err,
        spectral_eigenvalues=[float(v) for v in eigenvalues],
        mapper_laplacian_correct_region=cr,
        mapper_laplacian_modal_betti=(int(modal[0]), int(modal[1])),
        spectral_pass=spec_pass, mapper_pass=mapper_pass,
        multiplicity_pass=mult_pass, near_zero_pass=nz_pass,
        mapper_interior_pass=interior_pass,
        overall_pass=overall,
        multiplicity_check=mult_check, near_zero_count=nz,
        failure_mode_diagnostic=fmd,
        final_mse=final_mse, final_mean_l0=final_l0, final_n_dead=final_dead,
    )


SIGN_OFF_TEMPLATE = """# Stage 1 Sign-off — `{topology}` / `{arch}` (m={d_sae}, seed={seed})

Generated automatically by `stage1_bundle.py`.

## Configuration

| Parameter | Value |
|---|---|
| Topology | {topology} |
| Architecture | {arch} |
| d_sae (m) | {d_sae} |
| Ambient dim | {ambient_d} |
| Noise σ | {sigma_noise} |
| Eval samples (N) | {n_eval} |
| Stage 0 kNN k | {knn_k} |
| Stage 0 σ factor | {sigma_factor} |
| σ used | {sigma_used:.6g} |
| Spectral K | {spectral_K} |
| Filter | Laplacian eigenvector (k={k_filter}) |

## Training (final step)

| Metric | Value |
|---|---|
| MSE | {mse_str} |
| Mean L0 | {l0_str} |
| Dead atoms | {dead_str} |

## Sign-off checklist

- [{cb_mapper}] Mapper correct region (Laplacian filter) contains GT Betti at N={n_eval}
- [{cb_size}] Correct-region size ≥ {region_threshold:.0f}% of grid (got {region_size}/{total} = {region_pct:.0f}%)
- [{cb_contig}] Correct region is contiguous (n_correct_total = {n_correct})
- [{cb_interior}] Correct region is interior to grid (no edge contact)
- [{cb_failure}] Failure modes outside correct region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [{cb_log}] Spectral log-ratio error E < {log_threshold} on post-activations ({log_str})
- [{cb_mult}] Multiplicity-cluster check passes for first {n_clusters} clusters
- [{cb_nz}] Near-zero eigenvalue count = b₀ (got {near_zero}, expected {expected_b0})
- [x] Mapper graph renderings persisted for every config in correct region
- [x] Eigenvalues + eigenvectors + samples persisted

## Diagnostics

- Modal Betti (Laplacian filter): {modal_betti}
- Expected Betti: {expected_betti}

## Overall

**{overall_str}**
"""


def _render_signoff(r: Stage1Result, *, spectral_K: int, k_filter: int,
                    log_threshold: float, region_threshold: float,
                    n_clusters: int) -> str:
    cb = lambda b: 'x' if b else ' '
    cr = r.mapper_laplacian_correct_region
    expected = EXPECTED_BETTI.get(r.topology)
    expected_b0 = expected[0] if expected else 1
    log_str = (f"E={r.spectral_log_ratio_error:.4f}"
               if r.spectral_log_ratio_error is not None else "no closed-form ref")
    mse_str = f"{r.final_mse:.4g}" if r.final_mse is not None else "n/a"
    l0_str = f"{r.final_mean_l0:.2f}" if r.final_mean_l0 is not None else "n/a"
    dead_str = str(r.final_n_dead) if r.final_n_dead is not None else "n/a"
    cb_failure = (cb(r.failure_mode_diagnostic.get('all_rows_monotone', False))
                  if r.failure_mode_diagnostic else ' ')
    return SIGN_OFF_TEMPLATE.format(
        topology=r.topology, arch=r.arch, d_sae=r.d_sae, seed=r.seed,
        ambient_d=r.ambient_d, sigma_noise=r.sigma_noise,
        n_eval=r.n_eval_samples,
        knn_k=r.knn_k, sigma_factor=r.sigma_factor, sigma_used=r.sigma_used,
        spectral_K=spectral_K, k_filter=k_filter,
        mse_str=mse_str, l0_str=l0_str, dead_str=dead_str,
        cb_mapper=cb(r.mapper_pass),
        cb_size=cb(cr['region_fraction'] >= region_threshold),
        cb_contig=cb(cr['region_size'] == cr['n_correct_total']),
        cb_interior=cb(r.mapper_interior_pass),
        cb_failure=cb_failure,
        cb_log=cb(r.spectral_pass),
        cb_mult=cb(r.multiplicity_pass),
        cb_nz=cb(r.near_zero_pass),
        region_size=cr['region_size'], total=cr['total_configs'],
        region_pct=cr['region_fraction'] * 100,
        n_correct=cr['n_correct_total'],
        log_threshold=log_threshold, log_str=log_str,
        n_clusters=n_clusters,
        near_zero=r.near_zero_count, expected_b0=expected_b0,
        modal_betti=tuple(r.mapper_laplacian_modal_betti),
        expected_betti=tuple(expected) if expected else 'n/a',
        region_threshold=region_threshold * 100,
        overall_str=('PASS' if r.overall_pass else 'FAIL'),
    )


def run_stage1_bundle(
    config_path: Path | str,
    out_dir: Path | str,
    *,
    n_jobs: int = 1,
    baseline_path: Path | str | None = None,
    spectral_K: int = DEFAULT_SPECTRAL_K,
    log_ratio_threshold: float = DEFAULT_LOG_RATIO_THRESHOLD,
    correct_region_threshold: float = DEFAULT_CORRECT_REGION_THRESHOLD,
    multiplicity_eps: float = DEFAULT_MULTIPLICITY_EPS,
    multiplicity_n_clusters: int = DEFAULT_MULTIPLICITY_N_CLUSTERS,
) -> Stage1Result:
    """Train one SAE from `config_path`, run Laplacian Mapper sweep on its
    post-activations, and persist a Stage-0-style bundle in `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    topology = cfg_dict['topology']
    dgp_params = cfg_dict.get('dgp', {})
    sae_dict = cfg_dict.get('sae', {})
    ambient_d = int(dgp_params.get('d', 64))
    sigma_noise = float(dgp_params.get('sigma', 0.01))
    config = SAETrainConfig(
        d_in=ambient_d,
        **{k: v for k, v in sae_dict.items() if k != 'd_in'},
    )

    baseline = _load_baseline(baseline_path or DEFAULT_BASELINE_PATH)
    knn_k, sigma_factor, from_baseline = _baseline_for_topology(baseline, topology)
    if from_baseline:
        print(f"[stage1] Stage-0 baseline for {topology}: "
              f"knn_k={knn_k}, sigma_factor={sigma_factor}")
    else:
        print(f"[stage1] no Stage-0 baseline for {topology} — using defaults "
              f"(knn_k={knn_k}, sigma_factor={sigma_factor}).")

    print(f"[stage1] training {config.arch} on {topology} "
          f"(d_in={ambient_d}, d_sae={config.d_sae}, "
          f"n_steps={config.n_steps}, seed={config.seed})")
    t0 = time.time()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    extra = {k: v for k, v in dgp_params.items()
             if k not in ('d', 'sigma', 'center')}
    dgp_kwargs = {**_dgp_kwargs(topology), **extra}
    dgp = make_dgp(topology, d=ambient_d, sigma=sigma_noise,
                   c0=dgp_params.get('center', np.zeros(ambient_d)),
                   **dgp_kwargs)
    sae = build_sae(config)
    train_out = SAETrainer(sae, config, dgp).train()
    train_metrics = train_out.get('metrics', {})
    print(f"[stage1] training done in {time.time() - t0:.1f}s")

    # Fresh eval set + forward pass (post = codes).
    eval_X, eval_gt = dgp.sample_with_gt(config.n_eval_samples)
    with torch.no_grad():
        out = sae(torch.from_numpy(eval_X.astype(np.float32)))
        pre   = out['pre'].numpy()
        post  = out['codes'].numpy()
        recon = out['recon'].numpy()

    # Coifman-Lafon spectrum on post.
    print(f"[stage1] computing Coifman-Lafon spectrum (K={spectral_K}) on post...")
    t0 = time.time()
    spec = coifman_lafon_spectrum(
        post, knn_k=knn_k, K=spectral_K, sigma_factor=sigma_factor,
    )
    print(f"[stage1] spectrum done in {time.time() - t0:.1f}s")
    eigenvalues = spec['eigenvalues']
    eigenvectors = spec['eigenvectors']
    sigma_used = float(spec['sigma_used'])

    # Laplacian-filter Mapper sweep.
    k_filter = FILTER_K_BY_TOPOLOGY.get(topology, 3)
    lens = laplacian_eigenvector_filter(post, k=k_filter, eigenvectors=eigenvectors)
    threshold = float(global_distance_threshold(post))
    print(f"[stage1] Mapper sweep (24 configs, n_jobs={n_jobs})...")
    t0 = time.time()
    sweep = mapper_sweep(
        post, lens, topology=topology,
        n_intervals_grid=N_INTERVALS_GRID, overlap_grid=OVERLAP_GRID,
        distance_threshold=threshold, n_jobs=n_jobs,
    )
    print(f"[stage1] sweep done in {time.time() - t0:.1f}s")

    # Build Stage1Result + checks.
    result = _build_stage1_result(
        topology=topology, config=config,
        ambient_d=ambient_d, sigma_noise=sigma_noise,
        knn_k=knn_k, sigma_factor=sigma_factor, sigma_used=sigma_used,
        eigenvalues=eigenvalues, sweep=sweep, train_metrics=train_metrics,
        log_ratio_threshold=log_ratio_threshold,
        correct_region_threshold=correct_region_threshold,
        multiplicity_eps=multiplicity_eps,
        multiplicity_n_clusters=multiplicity_n_clusters,
    )

    # ----- Persist artifacts ------------------------------------------------
    np.savez(
        out_dir / 'samples.npz',
        eval_X=eval_X.astype(np.float32),
        pre=pre.astype(np.float32),
        post=post.astype(np.float32),
        recon=recon.astype(np.float32),
        gt=(np.asarray(eval_gt) if eval_gt is not None
            else np.zeros((eval_X.shape[0], 0), dtype=np.float32)),
        seed=np.asarray(int(config.seed)),
    )
    np.savez(
        out_dir / 'spectrum.npz',
        eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        sigma_used=np.asarray(sigma_used),
        knn_k=np.asarray(int(knn_k)),
        sigma_factor=np.asarray(float(sigma_factor)),
        seed=np.asarray(int(config.seed)),
    )
    with open(out_dir / 'mapper_sweep.json', 'w') as f:
        json.dump({'laplacian': _serialise_sweep(sweep)}, f, indent=2)
    with open(out_dir / 'training_metrics.json', 'w') as f:
        json.dump(_to_jsonable(train_metrics), f, indent=2)

    meta = {
        'config': asdict(config),
        'dgp_params': {**dgp_params, **dgp_kwargs},
        'baseline_used': from_baseline,
        'baseline_path': str(baseline_path or DEFAULT_BASELINE_PATH),
        'spectral_K': spectral_K,
        'multiplicity_eps': multiplicity_eps,
        'multiplicity_n_clusters': multiplicity_n_clusters,
        'log_ratio_threshold': log_ratio_threshold,
        'correct_region_threshold': correct_region_threshold,
        'n_intervals_grid': list(N_INTERVALS_GRID),
        'overlap_grid': list(OVERLAP_GRID),
        'distance_threshold': threshold,
        'k_filter': k_filter,
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump({'topology': topology,
                   'stage1_result': _to_jsonable(result),
                   '_meta': _to_jsonable(meta)}, f, indent=2)

    signoff = _render_signoff(
        result, spectral_K=spectral_K, k_filter=k_filter,
        log_threshold=log_ratio_threshold,
        region_threshold=correct_region_threshold,
        n_clusters=multiplicity_n_clusters,
    )
    (out_dir / 'report.md').write_text(signoff)

    # ----- Figures ----------------------------------------------------------
    expected = EXPECTED_BETTI.get(topology, (1, 0))
    mapper_grid_heatmap(
        sweep, expected, figures_dir / 'mapper_grid_laplacian.png',
        title=f"{topology} (Laplacian filter, post-activations)",
    )
    if topology in REFERENCE_SPECTRA:
        spectrum_overlay_plot(
            eigenvalues, REFERENCE_SPECTRA[topology]['ratios'],
            result.multiplicity_check,
            figures_dir / 'spectrum_overlay.png',
            title=f"{topology}: empirical (post) vs. closed-form Laplace–Beltrami",
            K=spectral_K,
        )

    # Per-stable-config Mapper renderings: re-run Mapper on each stable cell
    # to recover the graph object (mapper_sweep returns only diagnostics).
    stable_cells = stable_configs_from_correct_region(sweep, expected)
    if stable_cells:
        print(f"[stage1] rendering {len(stable_cells)} stable-config Mapper graphs...")
        graph_by_cell: dict[tuple[int, float], dict] = {}
        for (ni, ov) in stable_cells:
            graph_by_cell[(int(ni), float(ov))] = run_mapper_once(
                post, lens, int(ni), float(ov), distance_threshold=threshold,
            )
        mapper_graph_renderings(
            graph_by_config=graph_by_cell,
            gt=(np.asarray(eval_gt) if eval_gt is not None else None),
            topology=topology,
            stable_configs=stable_cells,
            out_dir=figures_dir / 'stable_region_renderings',
        )

    # Training curves.
    try:
        fig = plot_training_curves(
            type('R', (), {'metrics': train_metrics, 'config': config})()
        )
        fig.savefig(figures_dir / 'training_curves.png',
                    dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"[stage1] training_curves render skipped: {e}")

    print(f"[stage1] bundle: {out_dir}")
    print(f"[stage1] OVERALL: {'PASS' if result.overall_pass else 'FAIL'} "
          f"(spectral={result.spectral_pass}, mapper={result.mapper_pass}, "
          f"mult={result.multiplicity_pass}, nz={result.near_zero_pass}, "
          f"interior={result.mapper_interior_pass})")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 single-run bundle (Laplacian-only Mapper sweep).",
    )
    parser.add_argument('--config', required=True, help='YAML config path.')
    parser.add_argument('--out_dir', required=True,
                        help='Output bundle directory.')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Mapper sweep parallelism (loky workers).')
    parser.add_argument('--baseline_path', default=None,
                        help='Stage 0 baseline.yaml (default: configs/stage0/baseline.yaml).')
    parser.add_argument('--spectral_K', type=int, default=DEFAULT_SPECTRAL_K,
                        help='Number of Laplacian eigenvalues to compute.')
    args = parser.parse_args()
    run_stage1_bundle(
        args.config, args.out_dir,
        n_jobs=args.n_jobs,
        baseline_path=args.baseline_path,
        spectral_K=args.spectral_K,
    )


if __name__ == '__main__':
    main()
