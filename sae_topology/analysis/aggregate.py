"""Results aggregator + plotter.

Walks a results/ directory tree, loads the per-run JSON / NPY artifacts
emitted by `experiments.runner.ExperimentResult.save`, builds a summary
DataFrame, and produces figures: summary table, TopK phase-transition,
per-run Mapper graph diagnostics, per-run spectrum overlay.

Library API:
    df = load_results(results_dir)
    summary_table_figure(df) -> matplotlib.Figure
    topk_phase_transition_figure(df, manifold) -> matplotlib.Figure
    mapper_diagnostic_grid(run_dir, expected_betti=None) -> matplotlib.Figure
    spectrum_overlay_figure(run_dir) -> matplotlib.Figure

CLI:
    python -m sae_topology.analysis.aggregate \\
        --results_dir results/stage1 --out figures/stage1/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sae_topology.spectral import REFERENCE_SPECTRA
from sae_topology.analysis.plot import (
    plot_mapper_graph,
    plot_spectrum_vs_theory,
)


def load_results(
    results_dir: Path | str,
    glob: str = "*/result.json",
) -> pd.DataFrame:
    """Walk results_dir and load every result.json found.

    Returns a DataFrame with one row per run; columns include topology, arch,
    d_sae, k (TopK), seed, betti_lap, betti_gt, betti_pca,
    correct_region_fraction_lap, log_ratio_error, mse, mean_l0, run_dir,
    timestamp.
    """
    results_dir = Path(results_dir)
    rows = []
    for path in sorted(results_dir.glob(glob)):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        run_dir = path.parent
        cfg = d.get('config', {})
        mapper_summary_path = run_dir / 'mapper.json'
        cr_lap = None
        modal_lap = None
        if mapper_summary_path.exists():
            try:
                with open(mapper_summary_path) as f:
                    mp = json.load(f)
                lap_summary = mp.get('laplacian', {}).get('_summary', {})
                cr_lap = lap_summary.get('correct_region_fraction')
                modal_lap = lap_summary.get('modal_betti')
            except Exception:
                pass
        rows.append({
            'topology':   d.get('topology'),
            'arch':       d.get('arch'),
            'd_sae':      d.get('d_sae'),
            'k':          cfg.get('k'),
            'seed':       d.get('seed'),
            'timestamp':  d.get('timestamp'),
            'betti_lap':  tuple(d.get('headline_betti_laplacian') or []),
            'betti_gt':   tuple(d.get('headline_betti_gt') or []) if d.get('headline_betti_gt') else None,
            'betti_pca':  tuple(d.get('headline_betti_pca') or []) if d.get('headline_betti_pca') else None,
            'correct_region_fraction_lap': cr_lap,
            'modal_betti_lap': tuple(modal_lap) if modal_lap else None,
            'log_ratio_error': d.get('headline_log_ratio_error'),
            'mse':           d.get('final_mse'),
            'mean_l0':       d.get('final_mean_l0'),
            'l1_coeff':      cfg.get('l1_coeff'),
            'run_dir':       str(run_dir),
        })
    if not rows:
        return pd.DataFrame(columns=[
            'topology', 'arch', 'd_sae', 'k', 'seed', 'timestamp',
            'betti_lap', 'betti_gt', 'betti_pca',
            'correct_region_fraction_lap', 'log_ratio_error',
            'mse', 'mean_l0', 'run_dir',
        ])
    return pd.DataFrame(rows)


def summary_table_figure(df: pd.DataFrame) -> plt.Figure:
    """Render a summary DataFrame as a matplotlib table figure."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.text(0.5, 0.5, "No results found.", ha='center', va='center')
        ax.set_axis_off()
        return fig

    cols = ['topology', 'arch', 'd_sae', 'k', 'seed',
            'betti_lap', 'correct_region_fraction_lap',
            'log_ratio_error', 'mse', 'mean_l0']
    show = df[cols].copy()
    if 'correct_region_fraction_lap' in show.columns:
        show['correct_region_fraction_lap'] = show['correct_region_fraction_lap'].apply(
            lambda v: f"{v*100:.0f}%" if pd.notnull(v) else '-'
        )
    if 'log_ratio_error' in show.columns:
        show['log_ratio_error'] = show['log_ratio_error'].apply(
            lambda v: f"{v:.4f}" if pd.notnull(v) and np.isfinite(v) else 'inf' if pd.notnull(v) else '-'
        )
    if 'mse' in show.columns:
        show['mse'] = show['mse'].apply(
            lambda v: f"{v:.5f}" if pd.notnull(v) else '-'
        )
    if 'mean_l0' in show.columns:
        show['mean_l0'] = show['mean_l0'].apply(
            lambda v: f"{v:.2f}" if pd.notnull(v) else '-'
        )

    n_rows = len(show)
    fig, ax = plt.subplots(figsize=(min(16, 2 + 1.4 * len(cols)),
                                     0.5 + 0.3 * (n_rows + 1)))
    ax.set_axis_off()
    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.tight_layout()
    return fig


def topk_phase_transition_figure(
    df: pd.DataFrame,
    manifold: str,
    expected_d: Optional[int] = None,
) -> plt.Figure:
    """Plot correct-region-fraction and log-ratio-error vs TopK k for a manifold.

    Mean across seeds with min/max error band. If `expected_d` is given,
    draw a vertical dashed line at k = d+1 (H1 prediction).
    """
    sub = df[(df['topology'] == manifold) & (df['arch'] == 'topk')].copy()
    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No TopK results for {manifold}.", ha='center', va='center')
        ax.set_axis_off()
        return fig
    sub['k'] = sub['k'].astype(int)
    grouped = sub.groupby('k').agg(
        cr_mean=('correct_region_fraction_lap', 'mean'),
        cr_min=('correct_region_fraction_lap', 'min'),
        cr_max=('correct_region_fraction_lap', 'max'),
        err_mean=('log_ratio_error', 'mean'),
        err_min=('log_ratio_error', 'min'),
        err_max=('log_ratio_error', 'max'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(grouped['k'], grouped['cr_mean'], 'o-', color='steelblue', label='mean')
    ax.fill_between(grouped['k'], grouped['cr_min'], grouped['cr_max'],
                     alpha=0.25, color='steelblue', label='seed range')
    if expected_d is not None:
        ax.axvline(expected_d + 1, ls='--', color='firebrick',
                   label=f'H1: $k = d+1 = {expected_d + 1}$')
    ax.set_xlabel('TopK $k$')
    ax.set_ylabel('Correct-region fraction (Laplacian)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{manifold}: Mapper recovery vs $k$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    err_mean_clean = grouped['err_mean'].replace([np.inf, -np.inf], np.nan)
    err_min_clean = grouped['err_min'].replace([np.inf, -np.inf], np.nan)
    err_max_clean = grouped['err_max'].replace([np.inf, -np.inf], np.nan)
    ax.plot(grouped['k'], err_mean_clean, 'o-', color='tomato', label='mean')
    ax.fill_between(grouped['k'], err_min_clean, err_max_clean,
                     alpha=0.25, color='tomato', label='seed range')
    if expected_d is not None:
        ax.axvline(expected_d + 1, ls='--', color='firebrick',
                   label=f'H1: $k = d+1 = {expected_d + 1}$')
    ax.set_xlabel('TopK $k$')
    ax.set_ylabel(r'Spectral $\mathcal{E}$ (log-ratio error)')
    ax.set_yscale('log')
    ax.set_title(f'{manifold}: spectral error vs $k$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def _node_color_for_topology(
    graph: dict,
    topology: str,
    gt_coords: Optional[np.ndarray],
    eval_inputs: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], str, str]:
    """Pick the most informative node-coloring per manifold per the
    advisor's recommendation:

      - S^1   : mean(theta) where theta = atan2(gt[:,1], gt[:,0]); cyclic colormap.
      - T^2   : mean(theta) (caller may also request mean(phi) separately); cyclic.
      - S^2   : RGB from mean(x, y, z) - invariant to the SO(3) gauge of the
                 ell=1 spherical-harmonic eigenspace.
      - else  : sign-pinned mean(phi_1) of the Laplacian on the *input*
                 samples (not Z), fallback when no GT is surfaced.
                 Distinct from the cover coordinate, so coloring is a
                 genuine cross-check.

    Returns (node_color_array, cmap_name, label).
    """
    from sae_topology.mapper import node_means, pin_eigvec_sign
    from sae_topology.spectral import coifman_lafon_spectrum

    if topology == 'circle' and gt_coords is not None:
        gt = np.asarray(gt_coords)
        theta = np.arctan2(gt[:, 1], gt[:, 0])
        return node_means(graph, theta), 'twilight', r'mean $\theta$'
    if topology == 'torus' and gt_coords is not None:
        gt = np.asarray(gt_coords)
        theta = np.arctan2(gt[:, 1], gt[:, 0])
        return node_means(graph, theta), 'twilight', r'mean $\theta$'
    if topology == 'sphere' and gt_coords is not None:
        gt = np.asarray(gt_coords)
        rgb_pts = (gt + 1.0) / 2.0
        return node_means(graph, rgb_pts), '', r'RGB = (x, y, z)'
    if topology == 'helix' and gt_coords is not None:
        gt = np.asarray(gt_coords)
        s = gt[:, 0] if gt.ndim == 2 else gt
        return node_means(graph, s), 'viridis', r'arc-length $s$'
    if eval_inputs is not None:
        spec = coifman_lafon_spectrum(np.asarray(eval_inputs), knn_k=15, K=2)
        phi1 = pin_eigvec_sign(spec['eigenvectors'][:, 1])
        return node_means(graph, phi1), 'viridis', r'mean $\varphi_1$ (input-space)'
    return None, 'viridis', 'uniform'


def mapper_diagnostic_grid(
    run_dir: Path | str,
    expected_betti: Optional[tuple] = None,
) -> plt.Figure:
    """Per-run Mapper diagnostic: re-run Mapper at three representative
    cover configs and plot the graphs side by side, using a manifold-aware
    coloring scheme. For T^2, returns a 2-row figure (theta row + phi row)
    so neither cycle is hidden.
    """
    from sae_topology.spectral import coifman_lafon_spectrum
    from sae_topology.mapper import (
        laplacian_eigenvector_filter, run_mapper_once,
        global_distance_threshold, node_means,
    )

    run_dir = Path(run_dir)
    post_path = run_dir / 'post.npy'
    if not post_path.exists():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"post.npy missing in {run_dir}",
                ha='center', va='center')
        ax.set_axis_off()
        return fig
    post = np.load(post_path)
    gt_path = run_dir / 'gt.npy'
    gt_coords = np.load(gt_path) if gt_path.exists() else None
    inputs_path = run_dir / 'eval_inputs.npy'
    eval_inputs = np.load(inputs_path) if inputs_path.exists() else None

    with open(run_dir / 'result.json') as f:
        meta = json.load(f)
    topology = meta.get('topology', '?')

    k_filter = {'circle': 3, 'torus': 4, 'sphere': 4,
                'figure_eight': 3, 'helix': 2}.get(topology, 3)
    spec = coifman_lafon_spectrum(post, knn_k=15, K=k_filter + 1)
    lens = laplacian_eigenvector_filter(post, k=k_filter, eigenvectors=spec['eigenvectors'])
    thresh = global_distance_threshold(post)

    configs = [(5, 0.25), (8, 0.35), (12, 0.50)]

    if topology == 'torus' and gt_coords is not None:
        gt = np.asarray(gt_coords)
        theta = np.arctan2(gt[:, 1], gt[:, 0])
        phi   = np.arctan2(gt[:, 3], gt[:, 2])
        rows = [
            (theta, 'twilight', r'mean $\theta$'),
            (phi,   'twilight', r'mean $\varphi$'),
        ]
        fig, axes = plt.subplots(2, len(configs),
                                 figsize=(5 * len(configs), 9))
        graphs = [run_mapper_once(post, lens, ni, ov, distance_threshold=thresh)
                  for ni, ov in configs]
        for r, (vals, cmap, label) in enumerate(rows):
            for c, ((ni, ov), graph) in enumerate(zip(configs, graphs)):
                ax = axes[r, c]
                plot_mapper_graph(
                    graph, node_color=node_means(graph, vals),
                    cmap=cmap, ax=ax,
                    title=(f"ni={ni}, ov={ov}" if r == 0 else f"{label}"
                           if c == 0 else ''),
                )
            axes[r, 0].set_ylabel(label, fontsize=11)
        fig.suptitle(
            f"{topology} / {meta.get('arch', '?')} m={meta.get('d_sae')} "
            f"seed={meta.get('seed')}: Mapper graphs (Laplacian filter), "
            f"colored by GT coordinates",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 5))
    if len(configs) == 1:
        axes = [axes]
    for ax, (ni, ov) in zip(axes, configs):
        graph = run_mapper_once(post, lens, ni, ov, distance_threshold=thresh)
        nc, cmap, label = _node_color_for_topology(graph, topology, gt_coords, eval_inputs)
        plot_mapper_graph(graph, node_color=nc, cmap=cmap or 'viridis',
                          ax=ax, title=f"ni={ni}, ov={ov}")
    fig.suptitle(
        f"{topology} / {meta.get('arch', '?')} m={meta.get('d_sae')} "
        f"seed={meta.get('seed')}: Mapper graphs (Laplacian filter), "
        f"colored by {label}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def spectrum_overlay_figure(run_dir: Path | str) -> plt.Figure:
    """Overlay the run's empirical Coifman-Lafon eigenvalues against the
    closed-form Laplace-Beltrami spectrum for that manifold."""
    run_dir = Path(run_dir)
    with open(run_dir / 'result.json') as f:
        meta = json.load(f)
    topology = meta.get('topology', '?')
    with open(run_dir / 'spectral.json') as f:
        spec = json.load(f)

    best_k = spec.get('_best_knn_k')
    entry = spec.get(str(best_k)) or spec.get(best_k)
    if entry is None:
        # Fall back to the first available
        entry = next((v for k, v in spec.items() if not k.startswith('_')), None)
    if entry is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No spectral data in {run_dir}",
                ha='center', va='center')
        ax.set_axis_off()
        return fig
    eigs = np.asarray(entry['eigenvalues'])

    if topology in REFERENCE_SPECTRA:
        ratios = np.asarray(REFERENCE_SPECTRA[topology]['ratios'])
    else:
        ratios = np.array([])  # figure_eight: no closed-form reference

    title = (f"{topology} / {meta.get('arch', '?')} "
             f"m={meta.get('d_sae')} seed={meta.get('seed')}  "
             f"(knn_k={best_k}, $\\mathcal{{E}}$={entry.get('log_ratio_error')})")
    if ratios.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eigs / max(eigs[1] if len(eigs) > 1 else 1, 1e-12), 'o-',
                 color='steelblue')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('eig idx')
        ax.set_ylabel(r'$\hat\lambda_i / \hat\lambda_1$')
        return fig
    return plot_spectrum_vs_theory(eigs, ratios, title=title, K=20)


def main():
    parser = argparse.ArgumentParser(description="Aggregate SAE-experiment results.")
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing run subdirs with result.json')
    parser.add_argument('--out', default='figures/',
                        help='Output directory for figures + summary.csv')
    parser.add_argument('--per_run', action='store_true',
                        help='Also emit per-run Mapper + spectrum figures.')
    parser.add_argument('--topk_manifolds', nargs='*', default=None,
                        help='Manifolds to draw TopK phase figures for; '
                             'auto-detected from data if omitted.')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} runs from {args.results_dir}")
    if df.empty:
        print("Nothing to plot. Exiting.")
        return

    csv_path = out_dir / 'summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Wrote summary CSV: {csv_path}")

    fig = summary_table_figure(df)
    fig.savefig(out_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote summary table figure")

    topk_df = df[df['arch'] == 'topk']
    if not topk_df.empty:
        manifolds = args.topk_manifolds or sorted(topk_df['topology'].dropna().unique())
        d_by_manifold = {'circle': 1, 'torus': 2, 'sphere': 2, 'figure_eight': 1, 'helix': 1}
        for m in manifolds:
            fig = topk_phase_transition_figure(df, m, expected_d=d_by_manifold.get(m))
            fig.savefig(out_dir / f'topk_phase_{m}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Wrote topk_phase_{m}.png")

    if args.per_run:
        for _, row in df.iterrows():
            run_dir = Path(row['run_dir'])
            stem = run_dir.name
            try:
                fig = mapper_diagnostic_grid(run_dir)
                fig.savefig(out_dir / f'mapper_{stem}.png', dpi=120, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  mapper_{stem}: {e}")
            try:
                fig = spectrum_overlay_figure(run_dir)
                fig.savefig(out_dir / f'spectrum_{stem}.png', dpi=120, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  spectrum_{stem}: {e}")
        print(f"Wrote per-run figures for {len(df)} runs")

    print(f"All artifacts in {out_dir}")


if __name__ == '__main__':
    main()
