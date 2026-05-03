"""Single-run experiment: DGP -> train SAE -> Mapper + spectral analysis -> save.

Replaces the original T1/T2 PH pipeline (deleted module sae_topology.ph)
with the spec section 7 pipeline:
  1. Train SAE on DGP samples.
  2. Compute pre-activations, post-activations, reconstructions on fresh
     eval samples.
  3. Mapper analysis on post-activations under three filters
     (Laplacian eigenvector / GT / PCA), 20-config hyperparameter sweep.
  4. Coifman-Lafon spectral analysis on post-activations,
     three knn_k values.
  5. Diagnostic re-run on pre-activations and reconstructions if Mapper
     or spectral analysis on post-activations fails.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from sae_topology.dgp import make_dgp
from sae_topology.mapper import (
    laplacian_eigenvector_filter,
    ground_truth_filter,
    pca_filter,
    mapper_sweep,
    stable_betti,
    correct_region,
)
from sae_topology.saes.architectures import build_sae
from sae_topology.saes.config import SAETrainConfig
from sae_topology.saes.trainer import SAETrainer
from sae_topology.spectral import (
    coifman_lafon_spectrum,
    log_ratio_error,
    multiplicity_check,
    near_zero_count,
    REFERENCE_SPECTRA,
    GROUND_TRUTH_BETTI,
)


DEFAULT_BASELINE_PATH = Path('configs/stage0/baseline.yaml')
DEFAULT_KNN_K = 25
DEFAULT_SIGMA_FACTOR = 1.0
KNN_K_GRID: tuple[int, ...] = (15, 25, 40)
SPECTRAL_K: int = 20

EXPECTED_BETTI = {
    'circle': (1, 1),
    'torus':  (1, 2),
    'sphere': (1, 0),
    'figure_eight': (1, 2),
    'helix':  (1, 0),
}


def _load_stage0_baseline(path: Path | str | None = None) -> dict:
    """Load configs/stage0/baseline.yaml if present, else {}.

    Returns mapping topology -> {knn_k, sigma_factor, ...}. Topologies not
    present in the file fall back to runner defaults at use time.
    """
    if path is None:
        path = DEFAULT_BASELINE_PATH
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml as _yaml
        with open(path) as f:
            data = _yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"  [runner] failed to load Stage-0 baseline at {path}: {e}")
        return {}


def _baseline_for_topology(baseline: dict, topology: str) -> tuple[int, float, bool]:
    """Pull (knn_k, sigma_factor, is_from_baseline) for the topology."""
    entry = baseline.get(topology)
    if not entry:
        return DEFAULT_KNN_K, DEFAULT_SIGMA_FACTOR, False
    return (
        int(entry.get('knn_k', DEFAULT_KNN_K)),
        float(entry.get('sigma_factor', DEFAULT_SIGMA_FACTOR)),
        True,
    )

FILTER_K_BY_TOPOLOGY: dict[str, int] = {
    'circle': 3,
    'torus': 4,
    'sphere': 4,
    'two_circles': 3,
    'figure_eight': 3,
    'points': 3,
    'helix': 2,
}

GROUND_TRUTH_FILTER_OK = {'circle', 'torus', 'sphere', 'helix'}
SPECTRAL_REFERENCE_OK  = {'circle', 'torus', 'sphere', 'helix'}


@dataclass
class ExperimentResult:
    """All outputs from a single SAE training + topology-recovery run."""

    topology: str
    arch:     str
    d_sae:    int
    seed:     int

    pre_activations:  np.ndarray   # (n_eval, d_sae)
    post_activations: np.ndarray   # (n_eval, d_sae)
    reconstructions:  np.ndarray   # (n_eval, d_in)
    eval_inputs:      np.ndarray   # (n_eval, d_in) — the raw DGP samples fed to the SAE

    mapper:   dict   # {filter_name: {(n_intervals, overlap): {b0, b1, ...}, '_summary': {...}}}
    spectral: dict   # {knn_k: {eigenvalues, log_ratio_error, ...}, '_best_knn_k': int}
    metrics:  dict   # training: step, mse, loss, mean_l0, n_dead
    config:   SAETrainConfig

    gt_coords: Optional[np.ndarray] = None  # intrinsic GT coords or None
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def save(self, results_dir: Path | str) -> Path:
        """Persist to results_dir/{timestamp}_{topology}_{arch}_m{d_sae}_s{seed}/.

        Files written:
          pre.npy, post.npy, recon.npy   activation arrays
          mapper.json                    per-filter / per-config Mapper diagnostics
          spectral.json                  per-knn_k spectral diagnostics
          result.json                    training metrics + config + headlines
        """
        run_name = (
            f"{self.timestamp}_{self.topology}_{self.arch}"
            f"_m{self.d_sae}_s{self.seed}"
        )
        out = Path(results_dir) / run_name
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "pre.npy", self.pre_activations)
        np.save(out / "post.npy", self.post_activations)
        np.save(out / "recon.npy", self.reconstructions)
        np.save(out / "eval_inputs.npy", self.eval_inputs)
        if self.gt_coords is not None:
            np.save(out / "gt.npy", self.gt_coords)

        with open(out / "mapper.json", "w") as f:
            json.dump(_jsonify(self.mapper), f, indent=2)
        with open(out / "spectral.json", "w") as f:
            json.dump(_jsonify(self.spectral), f, indent=2)

        scalar = {
            "topology":  self.topology,
            "arch":      self.arch,
            "d_sae":     self.d_sae,
            "seed":      self.seed,
            "timestamp": self.timestamp,
            "headline_betti_laplacian": _safe_get(
                self.mapper, ['laplacian', '_summary', 'betti']
            ),
            "headline_betti_gt": _safe_get(
                self.mapper, ['ground_truth', '_summary', 'betti']
            ),
            "headline_betti_pca": _safe_get(
                self.mapper, ['pca', '_summary', 'betti']
            ),
            "headline_log_ratio_error": _safe_get(
                self.spectral,
                [self.spectral.get('_best_knn_k'), 'log_ratio_error'],
            ),
            "final_mse":   self.metrics["mse"][-1] if self.metrics.get("mse") else None,
            "final_mean_l0": self.metrics["mean_l0"][-1]
                              if self.metrics.get("mean_l0") else None,
            "config":      asdict(self.config),
        }
        with open(out / "result.json", "w") as f:
            json.dump(_jsonify(scalar), f, indent=2)
        return out


def run_sae_experiment(
    topology: str,
    config: SAETrainConfig,
    dgp_params: dict,
    save_dir: Path | str | None = None,
    knn_k_grid: Sequence[int] = KNN_K_GRID,
    spectral_K: int = SPECTRAL_K,
    skip_training: bool = False,
    baseline_path: Path | str | None = None,
    n_jobs: int = 1,
) -> ExperimentResult:
    """Train an SAE on synthetic DGP data, then run Mapper + spectral analysis
    on its post-activations.

    Parameters
    ----------
    topology     : DGP topology name.
    config       : SAETrainConfig.
    dgp_params   : dict with 'd', 'sigma' (required); 'center' (optional);
                   topology-specific kwargs (e.g. 'major_radius').
    save_dir     : if given, call result.save(save_dir).
    knn_k_grid   : kNN values to sweep for spectral analysis.
    spectral_K   : number of eigenvalues to compute per kNN value.
    skip_training: if True, skip training (random-init negative control).
    n_jobs       : worker count for the Mapper-filter pool (Laplacian / GT /
                   PCA dispatched in parallel via joblib loky). 1 = serial.
                   Each worker pins BLAS to 1 thread for determinism and calls
                   mapper_sweep with n_jobs=1 (no nested loky pools).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    baseline = _load_stage0_baseline(baseline_path)
    knn_k_baseline, sigma_factor_baseline, from_baseline = _baseline_for_topology(
        baseline, topology
    )
    if from_baseline:
        print(f"  [runner] Stage-0 baseline for {topology}: "
              f"knn_k={knn_k_baseline}, sigma_factor={sigma_factor_baseline}")
    else:
        print(f"  [runner] no Stage-0 baseline for {topology}; using defaults "
              f"(knn_k={knn_k_baseline}, sigma_factor={sigma_factor_baseline}). "
              f"Run `python -m sae_topology.experiments.stage0_validate` first.")

    d     = dgp_params["d"]
    sigma = dgp_params.get("sigma", 0.05)
    c0    = dgp_params.get("center", np.zeros(d))
    extra = {k: v for k, v in dgp_params.items()
             if k not in ('d', 'sigma', 'center')}
    dgp   = make_dgp(topology, d=d, sigma=sigma, c0=c0, **extra)

    sae = build_sae(config)

    if skip_training:
        train_metrics: dict = {'step': [], 'mse': [], 'loss': [],
                               'mean_l0': [], 'n_dead': []}
    else:
        train_out = SAETrainer(sae, config, dgp).train()
        train_metrics = train_out['metrics']

    eval_X, eval_gt = dgp.sample_with_gt(config.n_eval_samples)
    with torch.no_grad():
        x_t = torch.from_numpy(eval_X.astype(np.float32))
        out = sae(x_t)
        pre  = out['pre'].numpy()
        post = out['codes'].numpy()
        recon = out['recon'].numpy()

    mapper_results = _run_mapper_all_filters(
        post, eval_gt, topology,
        knn_k=knn_k_baseline, sigma_factor=sigma_factor_baseline,
        n_jobs=n_jobs,
    )
    spectral_results = _run_spectral_sweep(
        post, topology, knn_k_grid=knn_k_grid, K=spectral_K,
        sigma_factor=sigma_factor_baseline,
        best_knn_k_hint=knn_k_baseline,
    )

    result = ExperimentResult(
        topology=topology,
        arch=config.arch,
        d_sae=config.d_sae,
        seed=config.seed,
        pre_activations=pre,
        post_activations=post,
        reconstructions=recon,
        eval_inputs=eval_X,
        gt_coords=eval_gt,
        mapper=mapper_results,
        spectral=spectral_results,
        metrics=train_metrics,
        config=config,
    )
    if save_dir is not None:
        result.save(save_dir)
    return result


def _run_one_filter(
    filter_kind: str, Z: np.ndarray, gt: Optional[np.ndarray],
    topology: str, k: int, expected: Optional[tuple],
    eigenvectors: Optional[np.ndarray],
) -> tuple[str, dict]:
    """Worker: run one filter's Mapper sweep and summarize. Pinned to a
    single BLAS thread so parallel and serial paths produce identical
    numerical results.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if filter_kind == 'laplacian':
            if eigenvectors is None:
                raise ValueError(
                    "filter_kind='laplacian' requires precomputed eigenvectors."
                )
            lens = laplacian_eigenvector_filter(Z, k=k, eigenvectors=eigenvectors)
        elif filter_kind == 'ground_truth':
            if gt is None:
                raise ValueError("filter_kind='ground_truth' requires `gt`.")
            lens = ground_truth_filter(gt)
        elif filter_kind == 'pca':
            lens = pca_filter(Z, k=k)
        else:
            raise ValueError(f"unknown filter_kind {filter_kind!r}")

        sweep = mapper_sweep(
            Z, lens, topology=topology, n_jobs=1,  # never nest loky pools
        )
        summary = _summarise_sweep(sweep, k=lens.shape[1], expected_betti=expected)
    return filter_kind, summary


def _run_mapper_all_filters(
    Z: np.ndarray, gt: Optional[np.ndarray], topology: str,
    knn_k: int = DEFAULT_KNN_K,
    sigma_factor: float = DEFAULT_SIGMA_FACTOR,
    n_jobs: int = 1,
) -> dict:
    """Run Mapper under all available filters; return per-filter results.

    With n_jobs > 1 the per-filter sweeps are dispatched in parallel via
    joblib loky. The Coifman-Lafon spectrum (needed only for the Laplacian
    filter) is computed once in the main process and the eigenvectors are
    shipped to the Lap worker through pickle (small: N x (k+1) x 8 bytes).
    """
    k = FILTER_K_BY_TOPOLOGY.get(topology, 3)
    expected = EXPECTED_BETTI.get(topology)

    spec = coifman_lafon_spectrum(
        Z, knn_k=knn_k, K=k + 1, sigma_factor=sigma_factor,
    )
    eigenvectors = spec['eigenvectors']

    # Build the per-filter job list in the canonical (laplacian, gt, pca)
    # order so the returned dict has the same insertion order whether
    # dispatched serially or in parallel.
    jobs: list[tuple] = [
        ('laplacian', Z, gt, topology, k, expected, eigenvectors),
    ]
    if gt is not None and topology in GROUND_TRUTH_FILTER_OK:
        jobs.append(('ground_truth', Z, gt, topology, k, expected, None))
    jobs.append(('pca', Z, gt, topology, k, expected, None))

    if n_jobs == 1 or len(jobs) <= 1:
        outputs = [_run_one_filter(*job) for job in jobs]
    else:
        from joblib import Parallel, delayed
        outputs = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_run_one_filter)(*job) for job in jobs
        )

    # Re-key into a dict; iteration order matches the jobs list, which
    # matches the original sequential-loop order.
    return {filter_kind: payload for (filter_kind, payload) in outputs}


def _summarise_sweep(sweep: dict, k: Optional[int] = None,
                     expected_betti: Optional[tuple] = None) -> dict:
    """Wrap raw sweep dict with a '_summary' entry containing the headline
    Betti and the fraction of configs that produced it.

    Stored sweep keys are tuples (n_intervals, overlap); JSON-serialised by
    _jsonify into 'ni{n}_ov{overlap}' strings.
    """
    b0, b1, frac = stable_betti(sweep)
    summary: dict = {
        'modal_betti': [b0, b1],
        'modal_fraction': frac,
        'n_configs': len(sweep),
        'filter_dim': k,
    }
    if expected_betti is not None:
        cr = correct_region(sweep, expected_betti)
        summary['expected_betti'] = list(expected_betti)
        summary['correct_region_fraction'] = cr['region_fraction']
        summary['correct_region_size'] = cr['region_size']
        summary['n_correct_total'] = cr['n_correct_total']
        # Headline Betti = expected if there's a correct region of any size;
        # otherwise modal (so failure cases show their actual modal fragility).
        summary['betti'] = list(expected_betti) if cr['region_size'] > 0 else [b0, b1]
    else:
        summary['betti'] = [b0, b1]
    out = {f"ni{ni}_ov{ov}": v for (ni, ov), v in sweep.items()}
    out['_summary'] = summary
    return out


def _run_spectral_sweep(
    Z: np.ndarray, topology: str,
    knn_k_grid: Sequence[int],
    K: int,
    sigma_factor: float = DEFAULT_SIGMA_FACTOR,
    best_knn_k_hint: Optional[int] = None,
) -> dict:
    """Run Coifman-Lafon spectral analysis at each knn_k in the grid.

    Returns a dict {knn_k: {...}, '_best_knn_k': int} where the best knn_k
    minimises log_ratio_error against the closed-form LB spectrum (when
    available - otherwise the lowest knn_k is reported as 'best').
    """
    has_reference = topology in SPECTRAL_REFERENCE_OK
    ref = REFERENCE_SPECTRA.get(topology) if has_reference else None
    results: dict = {}
    best_k = None
    best_err = float('inf')

    for kk in knn_k_grid:
        spec = coifman_lafon_spectrum(
            Z, knn_k=int(kk), K=K, sigma_factor=sigma_factor,
        )
        eigs = spec['eigenvalues']
        entry: dict = {
            'eigenvalues': eigs.tolist(),
            'sigma_used': float(spec['sigma_used']),
            'near_zero_count': near_zero_count(eigs, threshold=1e-4),
        }
        if has_reference and ref is not None:
            try:
                err = log_ratio_error(eigs, ref['ratios'])
            except Exception:
                err = float('inf')
            try:
                mults = multiplicity_check(eigs, ref['levels'][:6], eps=0.10)
                entry['multiplicity'] = {str(L): c for L, c in mults.items()}
            except Exception:
                entry['multiplicity'] = None
            entry['log_ratio_error'] = err
            if err < best_err:
                best_err = err
                best_k = int(kk)
        results[int(kk)] = entry

    if best_k is None:
        best_k = int(knn_k_grid[0])
    if best_knn_k_hint is not None and best_knn_k_hint in results:
        results['_best_knn_k'] = int(best_knn_k_hint)
        results['_min_error_knn_k'] = best_k
    else:
        results['_best_knn_k'] = best_k
    return results


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays + tuple keys to JSON-safe form."""
    if isinstance(obj, dict):
        return {(_keystr(k)): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _keystr(k: Any) -> str:
    if isinstance(k, tuple):
        return "_".join(str(x) for x in k)
    return str(k)


def _safe_get(d: dict, path: list) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main():
    """CLI: python -m sae_topology.experiments.runner --config <yaml>"""
    import argparse
    import yaml as _yaml
    parser = argparse.ArgumentParser(description="Run a single SAE experiment from a YAML config.")
    parser.add_argument('--config', required=True, help='YAML config path')
    parser.add_argument('--save_dir', default='results/stage1',
                        help='Directory to save results to')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Worker count for the per-filter Mapper joblib pool. '
                             '1 = serial. Each worker pins BLAS to 1 thread.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = _yaml.safe_load(f)
    topology = cfg_dict['topology']
    dgp_params = cfg_dict.get('dgp', {})
    sae_dict = cfg_dict.get('sae', {})
    config = SAETrainConfig(
        d_in=dgp_params.get('d', 64),
        **{k: v for k, v in sae_dict.items() if k != 'd_in'},
    )
    res = run_sae_experiment(
        topology, config, dgp_params, save_dir=args.save_dir, n_jobs=args.n_jobs,
    )
    print(f"\n=== {topology} / {config.arch} m={config.d_sae} ===")
    lap = res.mapper.get('laplacian', {}).get('_summary', {})
    err = res.spectral.get(res.spectral.get('_best_knn_k'), {}).get('log_ratio_error')
    print(f"  Laplacian Betti: {lap.get('betti')} "
          f"(stability {lap.get('stability_fraction', 0)*100:.0f}%)")
    print(f"  Spectral log-ratio error: {err}")


if __name__ == '__main__':
    main()
