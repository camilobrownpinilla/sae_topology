"""Width and architecture sweep utilities for Stages 3 and 4."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sae_topology.saes.config import SAETrainConfig
from .runner import ExperimentResult, run_sae_experiment


def _run_all(
    jobs: list[tuple],
    n_jobs: int,
    results_dir: Path | None,
) -> list[ExperimentResult]:
    """Execute (topology, config, dgp_params) tuples, optionally in parallel."""
    if n_jobs == 1:
        return [
            run_sae_experiment(topo, cfg, dgp_p, save_dir=results_dir)
            for topo, cfg, dgp_p in jobs
        ]

    try:
        from joblib import Parallel, delayed
    except ImportError as e:
        raise ImportError("Install joblib for n_jobs > 1: pip install joblib") from e

    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_sae_experiment)(topo, cfg, dgp_p, results_dir)
        for topo, cfg, dgp_p in jobs
    )


def width_sweep(
    topology: str = "circle",
    widths: list[int] | None = None,
    arch: str = "relu_l1",
    n_seeds: int = 3,
    base_config: SAETrainConfig | None = None,
    dgp_params: dict | None = None,
    n_jobs: int = 1,
    results_dir: Path | str | None = None,
) -> list[ExperimentResult]:
    """Sweep over SAE widths and seeds for a single topology and architecture.

    Parameters
    ----------
    topology    : DGP topology name
    widths      : list of d_sae values  (default: [16, 64, 256, 1024])
    arch        : 'relu_l1' | 'topk' | 'jumprelu'
    n_seeds     : number of random seeds per (width) combo
    base_config : SAETrainConfig template; d_in and d_sae are overridden.
                  Defaults to relu_l1, d_in=8 if not provided.
    dgp_params  : dict with 'd' and 'sigma' keys (default: d=8, sigma=0.05)
    n_jobs      : joblib parallelism (1 = sequential)
    results_dir : if given, each run saves its output here

    Returns
    -------
    list of ExperimentResult, length = len(widths) * n_seeds
    """
    if widths is None:
        widths = [16, 64, 256, 1024]
    if dgp_params is None:
        dgp_params = {"d": 8, "sigma": 0.05}
    if base_config is None:
        base_config = SAETrainConfig(arch=arch, d_in=dgp_params["d"], d_sae=64)

    if results_dir is not None:
        results_dir = Path(results_dir)

    jobs = []
    for m in widths:
        for seed in range(n_seeds):
            cfg = replace(base_config, arch=arch, d_in=dgp_params["d"],
                          d_sae=m, seed=seed)
            jobs.append((topology, cfg, dgp_params))

    return _run_all(jobs, n_jobs, results_dir)


def topk_k_sweep(
    topology: str = "circle",
    K_values: list[int] | None = None,
    n_seeds: int = 3,
    base_config: SAETrainConfig | None = None,
    dgp_params: dict | None = None,
    n_jobs: int = 1,
    results_dir: Path | str | None = None,
) -> list[ExperimentResult]:
    """Sweep over TopK k values, holding d_sae fixed.

    Drives the H1 phase-transition figure (spec section 6, line 217):
    for a d-manifold, recovery (correct Betti, low spectral error) is
    predicted to require k >= d + 1.

    Parameters
    ----------
    topology    : DGP topology name.
    K_values    : list of TopK k values  (default: [1, 2, 3, 4, 6, 8])
    n_seeds     : seeds per k value.
    base_config : template config; arch is forced to 'topk', d_in/d_sae kept,
                  k is overridden by the sweep, seed is overridden per run.
    dgp_params  : passed through to run_sae_experiment.
    n_jobs      : joblib parallelism.
    results_dir : if given, each run saves output here.
    """
    if K_values is None:
        K_values = [1, 2, 3, 4, 6, 8]
    if dgp_params is None:
        dgp_params = {"d": 64, "sigma": 0.01}
    if base_config is None:
        base_config = SAETrainConfig(arch="topk", d_in=dgp_params["d"], d_sae=64)

    if results_dir is not None:
        results_dir = Path(results_dir)

    jobs = []
    for k_val in K_values:
        for seed in range(n_seeds):
            cfg = replace(
                base_config,
                arch="topk",
                d_in=dgp_params["d"],
                k=k_val,
                seed=seed,
            )
            jobs.append((topology, cfg, dgp_params))

    return _run_all(jobs, n_jobs, results_dir)


def main():
    """CLI: python -m sae_topology.experiments.sweep --topk-k --manifold circle --K 1,2,3,4"""
    import argparse
    parser = argparse.ArgumentParser(description="Run an SAE sweep.")
    parser.add_argument('--topk-k', action='store_true',
                        help="Run topk_k_sweep (TopK k phase-transition sweep).")
    parser.add_argument('--manifold', default='circle',
                        help="Topology to sweep on.")
    parser.add_argument('--K', default='1,2,3,4,6,8',
                        help="Comma-separated TopK k values.")
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=30_000)
    parser.add_argument('--ambient_d', type=int, default=64)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--d_sae', type=int, default=64)
    parser.add_argument('--results_dir', default='results/topk_sweep')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    if not args.topk_k:
        parser.error("Currently only --topk-k is supported. "
                     "Use the Python API for width / architecture sweeps.")

    K_values = [int(k) for k in args.K.split(',')]
    base = SAETrainConfig(
        arch='topk', d_in=args.ambient_d, d_sae=args.d_sae,
        n_steps=args.n_steps,
    )
    dgp_params = {'d': args.ambient_d, 'sigma': args.sigma}
    if args.manifold == 'torus':
        dgp_params.update({'major_radius': 1.0, 'minor_radius': 1.0})
    elif args.manifold == 'sphere':
        dgp_params.update({'radius': 1.0})
    elif args.manifold == 'helix':
        dgp_params.update({'radius': 1.0, 'pitch': 0.5, 'n_turns': 4.0})

    print(f"TopK k sweep on {args.manifold}, K={K_values}, "
          f"n_seeds={args.n_seeds}, n_steps={args.n_steps}")
    results = topk_k_sweep(
        topology=args.manifold,
        K_values=K_values,
        n_seeds=args.n_seeds,
        base_config=base,
        dgp_params=dgp_params,
        n_jobs=args.n_jobs,
        results_dir=args.results_dir,
    )

    print(f"\n=== TopK-k sweep summary ===")
    for r in results:
        lap = r.mapper.get('laplacian', {}).get('_summary', {})
        err = r.spectral.get(r.spectral.get('_best_knn_k'), {}).get('log_ratio_error')
        print(f"  k={r.config.k} seed={r.seed}: "
              f"Betti={lap.get('betti')} "
              f"stab={lap.get('stability_fraction', 0)*100:.0f}%  E={err}")


def architecture_sweep(
    topologies: list[str] | None = None,
    archs: list[str] | None = None,
    widths: list[int] | None = None,
    n_seeds: int = 3,
    base_configs: dict[str, SAETrainConfig] | None = None,
    dgp_params: dict | None = None,
    n_jobs: int = 1,
    results_dir: Path | str | None = None,
) -> list[ExperimentResult]:
    """Full Stage 4 architecture sweep.

    Runs all combinations of topology × arch × width × seed.

    Parameters
    ----------
    topologies  : list of DGP topology names
                  (default: circle, two_circles, figure_eight, torus)
    archs       : list of arch names  (default: ['relu_l1', 'topk'])
    widths      : list of d_sae values  (default: [16, 64, 256, 1024])
    n_seeds     : seeds per combo  (default: 3)
    base_configs: dict mapping arch → SAETrainConfig template.
                  d_in, d_sae, seed, arch are overridden per run.
                  Missing keys fall back to SAETrainConfig defaults.
    dgp_params  : dict with 'd' and 'sigma' (default: d=8, sigma=0.05)
    n_jobs      : joblib parallelism
    results_dir : if given, each run saves output here

    Returns
    -------
    list of ExperimentResult, length ≈ len(topologies)*len(archs)*len(widths)*n_seeds
    """
    if topologies is None:
        topologies = ["circle", "two_circles", "figure_eight", "torus"]
    if archs is None:
        archs = ["relu_l1", "topk"]
    if widths is None:
        widths = [16, 64, 256, 1024]
    if dgp_params is None:
        dgp_params = {"d": 8, "sigma": 0.05}
    if base_configs is None:
        base_configs = {}

    if results_dir is not None:
        results_dir = Path(results_dir)

    jobs = []
    for topo in topologies:
        for arch in archs:
            tmpl = base_configs.get(
                arch,
                SAETrainConfig(arch=arch, d_in=dgp_params["d"], d_sae=64),
            )
            for m in widths:
                for seed in range(n_seeds):
                    cfg = replace(tmpl, arch=arch, d_in=dgp_params["d"],
                                  d_sae=m, seed=seed)
                    jobs.append((topo, cfg, dgp_params))

    return _run_all(jobs, n_jobs, results_dir)


if __name__ == '__main__':
    main()
