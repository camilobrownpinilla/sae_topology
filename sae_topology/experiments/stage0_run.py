"""Stage 0 full validation orchestrator.

End-to-end execution of stage0_tuning.md §10 sign-off checklist for $S^1$,
$T^2$, $S^2$ at $N=100{,}000$ in $\\mathbb{R}^{64}$:

  python -m sae_topology.experiments.stage0_run \\
      --topologies circle torus sphere \\
      --n_samples 100000 --n_jobs 56 \\
      --out_dir results/stage0

What it does
------------
1. Calls `run_stage0` (auto-tune + flat 216-job Stage B sweep). Captures
   the per-(topology, filter) sweep dicts AND the winners' eigenvectors via
   side-channel (avoids redundant eigsh / Mapper sweeps in step 4).
2. Per topology, in parallel:
   - Re-derive sample point cloud + GT for artifact persistence.
   - Run spectral cross-seed reproducibility (T^2 / S^2 only).
   - Persist samples / spectrum / mapper sweep / report.json.
   - Render Mapper-grid heatmap, spectrum overlay, and per-stable-config
     Mapper-graph PNGs (colored by GT coords).
   - Write per-topology sign-off markdown.
3. Write top-level `STAGE0_SUMMARY.md` with the per-manifold pass/fail table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

from sae_topology.spectral import REFERENCE_SPECTRA
from sae_topology.mapper import (
    N_INTERVALS_GRID,
    OVERLAP_GRID,
    laplacian_eigenvector_filter,
    run_mapper_once,
    global_distance_threshold,
)
from sae_topology.experiments.stage0_validate import (
    DEFAULT_KNN_K_GRID,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SIGMA_FACTOR_GRID,
    DEFAULT_STAGE0_TOPOLOGIES,
    EXPECTED_BETTI,
    FILTER_K_BY_TOPOLOGY,
    INNER_NI_GRID,
    Stage0Result,
    _sample_X_gt,
    run_stage0,
)
from sae_topology.experiments.stage0_reproducibility import (
    REPRODUCIBILITY_TOPOLOGIES,
    run_spectral_reproducibility,
)
from sae_topology.experiments.stage0_artifacts import (
    persist_topology_bundle,
    render_signoff_md,
)
from sae_topology.experiments.stage0_plots import (
    mapper_grid_heatmap,
    mapper_graph_renderings,
    spectrum_overlay_plot,
    stable_configs_from_correct_region,
)


def _post_process_topology(
    topology: str,
    n_samples: int,
    ambient_d: int,
    sigma_data: float,
    seed: int,
    knn_k: int,
    sigma_factor: float,
    k_filter: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sigma_used: float,
    sweep_lap: dict,
    sweep_gt: Optional[dict],
    sweep_pca: Optional[dict],
    stage0_result: Stage0Result,
    out_dir: Path,
    spectral_K: int,
    multiplicity_eps: float,
    multiplicity_n_clusters: int,
    repro_seeds: tuple[int, int],
    log_ratio_tolerance: float,
    meta: dict,
) -> dict:
    """Per-topology post-processing in one worker. BLAS pinned for
    determinism."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        t0 = time.time()
        out_dir = Path(out_dir)

        # 1. Re-derive X + gt for artifact persistence (cheap from seed).
        X, gt = _sample_X_gt(topology, n_samples, ambient_d, sigma_data, seed)

        # 2. Reproducibility check (T^2 / S^2 only).
        repro = None
        if topology in REPRODUCIBILITY_TOPOLOGIES:
            print(f"  [{topology}] running spectral reproducibility check ...",
                  flush=True)
            repro = run_spectral_reproducibility(
                topology=topology,
                n_samples=n_samples,
                ambient_d=ambient_d,
                sigma_data=sigma_data,
                knn_k=knn_k,
                sigma_factor=sigma_factor,
                K=spectral_K,
                seeds=repro_seeds,
                log_ratio_tolerance=log_ratio_tolerance,
                multiplicity_eps=multiplicity_eps,
                multiplicity_n_clusters=multiplicity_n_clusters,
            )

        # 3. Persist bundle (samples / spectrum / sweep / report.json).
        bundle_dir = persist_topology_bundle(
            topology=topology,
            out_dir=out_dir,
            X=X, gt=gt,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            sigma_used=sigma_used,
            knn_k=knn_k,
            sigma_factor=sigma_factor,
            seed=seed,
            sweep_lap=sweep_lap,
            sweep_gt=sweep_gt,
            sweep_pca=sweep_pca,
            stage0_result=stage0_result,
            reproducibility=repro,
            meta=meta,
        )
        figures_dir = bundle_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        expected = EXPECTED_BETTI.get(topology, None)

        # 4. Mapper grid heatmaps.
        if expected:
            mapper_grid_heatmap(
                sweep_lap, expected,
                figures_dir / 'mapper_grid_laplacian.png',
                title=f"{topology} — Laplacian filter",
            )
            if sweep_gt is not None:
                mapper_grid_heatmap(
                    sweep_gt, expected,
                    figures_dir / 'mapper_grid_gt.png',
                    title=f"{topology} — GT filter",
                )
            if sweep_pca is not None:
                mapper_grid_heatmap(
                    sweep_pca, expected,
                    figures_dir / 'mapper_grid_pca.png',
                    title=f"{topology} — PCA filter",
                )

        # 5. Spectrum overlay (only for topologies with closed-form ref).
        if topology in REFERENCE_SPECTRA:
            ref = REFERENCE_SPECTRA[topology]
            spectrum_overlay_plot(
                eigenvalues, ref['ratios'],
                multiplicity_check=stage0_result.multiplicity_check,
                out_path=figures_dir / 'spectrum_overlay.png',
                title=f"{topology}: empirical vs. closed-form Laplace–Beltrami",
                K=spectral_K,
            )

        # 6. Per-stable-config Mapper-graph renderings (PNGs, GT-colored).
        stable_cells = stable_configs_from_correct_region(
            sweep_lap, expected,
        ) if expected else []
        if stable_cells:
            print(f"  [{topology}] rendering {len(stable_cells)} stable-config "
                  f"Mapper graphs ...", flush=True)
            lens = laplacian_eigenvector_filter(
                X, k=k_filter, eigenvectors=eigenvectors,
            )
            thresh = float(global_distance_threshold(X))
            graph_by_cell: dict[tuple[int, float], dict] = {}
            for (ni, ov) in stable_cells:
                graph_by_cell[(int(ni), float(ov))] = run_mapper_once(
                    X, lens, int(ni), float(ov), distance_threshold=thresh,
                )
            mapper_graph_renderings(
                graph_by_config=graph_by_cell,
                gt=gt,
                topology=topology,
                stable_configs=stable_cells,
                out_dir=figures_dir / 'stable_region_renderings',
            )

        # 7. Sign-off markdown.
        signoff_text = render_signoff_md(
            topology=topology,
            stage0_result=stage0_result,
            reproducibility=repro,
            n_clusters=multiplicity_n_clusters,
        )
        with open(bundle_dir / 'report.md', 'w') as f:
            f.write(signoff_text)

        wall = time.time() - t0
        print(f"  [{topology}] post-processing: {wall:.1f}s", flush=True)

        return {
            'topology': topology,
            'bundle_dir': str(bundle_dir),
            'overall_pass': bool(stage0_result.overall_pass),
            'reproducibility_pass': (repro['pass'] if repro else None),
            'wall': wall,
        }


def run_stage0_full(
    topologies: Sequence[str] = DEFAULT_STAGE0_TOPOLOGIES,
    out_dir: Path | str = Path('results/stage0'),
    n_samples_override: Optional[int] = None,
    ambient_d: int = 64,
    sigma: float = 0.01,
    seed: int = 0,
    n_jobs: int = 56,
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
    repro_seeds: tuple[int, int] = (0, 1),
    log_ratio_tolerance: float = 0.02,
) -> dict:
    """Top-level Stage 0 validation: run_stage0 + per-topology post-processing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples_map: Optional[dict[str, int]] = None
    if n_samples_override is not None:
        n_samples_map = {topo: int(n_samples_override) for topo in topologies}

    # ----- Auto-tune + flat Stage B (with sweep + winner capture) ----------
    captured_sweeps: dict[str, dict[str, dict]] = {}
    captured_winners: dict[str, dict] = {}

    t0 = time.time()
    print(f"\n=== STAGE 0 FULL: {list(topologies)} at "
          f"N={n_samples_override or 'default'} (n_jobs={n_jobs}) ===",
          flush=True)
    stage0_results = run_stage0(
        topologies=topologies,
        out_yaml=out_dir / 'baseline.yaml',
        out_json=out_dir / 'baseline.json',
        n_jobs=n_jobs,
        ambient_d=ambient_d,
        sigma=sigma,
        seed=seed,
        knn_k_grid=knn_k_grid,
        sigma_factor_grid=sigma_factor_grid,
        n_intervals_grid=n_intervals_grid,
        overlap_grid=overlap_grid,
        inner_n_intervals_grid=inner_n_intervals_grid,
        spectral_K=spectral_K,
        log_ratio_threshold=log_ratio_threshold,
        correct_region_threshold=correct_region_threshold,
        multiplicity_eps=multiplicity_eps,
        multiplicity_n_clusters=multiplicity_n_clusters,
        n_samples_override=n_samples_map,
        capture_sweeps_into=captured_sweeps,
        capture_winners_into=captured_winners,
    )
    autotune_wall = time.time() - t0
    print(f"\n=== Auto-tune + flat Stage B done: {autotune_wall:.1f}s ===",
          flush=True)

    meta = {
        'n_samples': n_samples_override or DEFAULT_SAMPLE_SIZES.get(
            topologies[0] if topologies else 'circle', 100_000,
        ),
        'ambient_d': ambient_d,
        'sigma_noise': sigma,
        'seed': seed,
        'n_intervals_grid': list(n_intervals_grid),
        'overlap_grid': list(overlap_grid),
        'knn_k_grid': list(knn_k_grid),
        'sigma_factor_grid': list(sigma_factor_grid),
        'spectral_K': spectral_K,
        'multiplicity_eps': multiplicity_eps,
        'multiplicity_n_clusters': multiplicity_n_clusters,
        'sigma_rule': 'median_heuristic',
        'laplacian': 'coifman_lafon_alpha1',
        'cluster_rule': 'single_linkage_global_5x_median_knn',
        'random_lift': f'random_orthonormal_R{ambient_d}',
    }

    # ----- Build per-topology post-processing jobs -------------------------
    bundle_jobs = []
    for topo in topologies:
        r = stage0_results[topo]
        win = captured_winners.get(topo, {})
        eigvals = np.asarray(win.get('spectral_eigenvalues', r.spectral_eigenvalues))
        eigvecs = np.asarray(win['eigenvectors'])  # required
        sigma_used = float(win.get('sigma_used', r.sigma_used))
        sweep_lap = captured_sweeps.get(topo, {}).get('lap', {})
        sweep_gt = captured_sweeps.get(topo, {}).get('gt', None)
        sweep_pca = captured_sweeps.get(topo, {}).get('pca', None)
        bundle_jobs.append((
            topo,
            r.n_samples, ambient_d, sigma, seed,
            r.knn_k, r.sigma_factor, FILTER_K_BY_TOPOLOGY[topo],
            eigvals, eigvecs, sigma_used,
            sweep_lap, sweep_gt, sweep_pca,
            r,
            out_dir,
            spectral_K, multiplicity_eps, multiplicity_n_clusters,
            tuple(repro_seeds), log_ratio_tolerance,
            meta,
        ))

    n_post_jobs = min(len(bundle_jobs), max(1, n_jobs))
    print(f"\n=== Post-processing: {len(bundle_jobs)} topologies "
          f"(n_jobs={n_post_jobs}) ===", flush=True)
    if n_post_jobs == 1 or len(bundle_jobs) <= 1:
        bundle_outputs = [_post_process_topology(*j) for j in bundle_jobs]
    else:
        bundle_outputs = Parallel(n_jobs=n_post_jobs, backend='loky')(
            delayed(_post_process_topology)(*j) for j in bundle_jobs
        )

    # ----- Top-level summary -----------------------------------------------
    total_wall = time.time() - t0
    overall_pass_all = True
    summary_lines = [
        "# Stage 0 Summary",
        "",
        f"- Total wall-clock: **{total_wall:.1f}s**",
        f"- Auto-tune + flat Stage B: {autotune_wall:.1f}s",
        f"- Post-processing: {total_wall - autotune_wall:.1f}s",
        f"- N = {meta['n_samples']}, ambient_d = {ambient_d}, σ = {sigma}, seed = {seed}",
        "",
        "| Topology | Overall | Reproducibility (T²/S²) | Bundle |",
        "|---|---|---|---|",
    ]
    for out in bundle_outputs:
        passed = bool(out['overall_pass'])
        repro_pass = out['reproducibility_pass']
        overall_pass_all = overall_pass_all and passed
        rep_str = ('PASS' if repro_pass else
                   ('FAIL' if repro_pass is False else 'n/a'))
        bundle_rel = Path(out['bundle_dir']).resolve().relative_to(
            out_dir.resolve().parent
        )
        summary_lines.append(
            f"| {out['topology']} "
            f"| {'PASS' if passed else 'FAIL'} "
            f"| {rep_str} "
            f"| `{bundle_rel}` |"
        )
    summary_lines.append("")
    summary_lines.append(f"\n**Overall: {'PASS' if overall_pass_all else 'FAIL'}**")
    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / 'STAGE0_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\n=== Total wall-clock: {total_wall:.1f}s ===")
    print(f"Wrote {summary_path}")

    return {
        'stage0_results': stage0_results,
        'bundle_outputs': bundle_outputs,
        'summary_path': str(summary_path),
        'wall_total': total_wall,
        'wall_autotune': autotune_wall,
        'meta': meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0 full validation execution",
    )
    parser.add_argument('--topologies', nargs='+',
                        default=list(DEFAULT_STAGE0_TOPOLOGIES))
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Override per-topology N (default: 100k).')
    parser.add_argument('--ambient_d', type=int, default=64)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='results/stage0')
    parser.add_argument('--n_jobs', type=int, default=56)
    args = parser.parse_args()

    run_stage0_full(
        topologies=args.topologies,
        out_dir=Path(args.out_dir),
        n_samples_override=args.n_samples,
        ambient_d=args.ambient_d,
        sigma=args.sigma,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()
