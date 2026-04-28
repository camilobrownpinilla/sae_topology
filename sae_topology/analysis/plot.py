"""Plotting helpers for the Mapper + Laplace-Beltrami pipeline."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA


# ─── DGP sanity check ──────────────────────────────────────────────────────


def plot_dgp_samples(topologies: list, make_dgp_fn, d: int = 64,
                     sigma: float = 0.01, N: int = 500,
                     seed: int = 42, pca_dim: int = 2) -> plt.Figure:
    """PCA projections of point clouds for each topology."""
    np.random.seed(seed)
    c0 = np.zeros(d)
    label_map = {
        'points': 'Points', 'circle': 'Circle', 'two_circles': 'Two Circles',
        'figure_eight': 'Figure 8', 'torus': 'Torus', 'sphere': 'Sphere',
    }
    if pca_dim == 3:
        fig, axes = plt.subplots(1, len(topologies),
                                 figsize=(4 * len(topologies), 4),
                                 subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(1, len(topologies),
                                 figsize=(4 * len(topologies), 4))
    if len(topologies) == 1:
        axes = [axes]

    for ax, topo in zip(axes, topologies):
        kw = {}
        if topo == 'torus':
            kw = {'major_radius': 1.0, 'minor_radius': 1.0}
        X = make_dgp_fn(topo, d, sigma, c0, **kw).sample(N)
        Xp = PCA(n_components=pca_dim).fit_transform(X)
        if pca_dim == 3:
            ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=4, alpha=0.6)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        else:
            ax.scatter(Xp[:, 0], Xp[:, 1], s=4, alpha=0.6)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_aspect('equal')
        ax.set_title(label_map.get(topo, topo))

    fig.suptitle(f'DGP Sanity Check — {pca_dim}D PCA Projections', fontsize=13)
    fig.tight_layout()
    return fig


# ─── Training curves ───────────────────────────────────────────────────────


def plot_training_curves(result) -> plt.Figure:
    """3-panel diagnostic from an ExperimentResult: MSE, mean L0, dead atoms."""
    m = result.metrics
    if not m.get('step'):
        raise ValueError("ExperimentResult has no training metrics (skip_training?)")
    steps   = np.array(m['step'])
    mse     = np.array(m['mse'])
    mean_l0 = np.array(m['mean_l0'])
    n_dead  = np.array(m['n_dead'])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    re = result.config.resample_every
    resample_steps = steps[steps % re == 0][1:]
    for ax in axes:
        for rs in resample_steps:
            ax.axvline(rs, color='gray', linestyle=':', lw=0.8, alpha=0.6)

    axes[0].plot(steps, mse, lw=1.5, color='steelblue')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('MSE')
    axes[0].set_title('Reconstruction MSE')

    axes[1].plot(steps, mean_l0, lw=1.5, color='tomato')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Mean L0')
    axes[1].set_title('Sparsity (mean active features)')

    axes[2].plot(steps, n_dead, lw=1.5, color='mediumseagreen')
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('# Dead atoms')
    axes[2].set_title('Dead atoms')

    fig.suptitle(
        f'Training — {result.topology} / {result.arch}, '
        f'm={result.d_sae}, seed={result.seed}',
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ─── Mapper graph ──────────────────────────────────────────────────────────


def plot_mapper_graph(
    graph: dict,
    node_color: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    layout_seed: int = 0,
    colorbar: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """networkx visualisation of a kmapper graph dict.

    `node_color` shape:
      - (n_nodes,)  -> scalar mapped through `cmap`. Use 'twilight' for
                       cyclic data (theta on S^1 / T^2).
      - (n_nodes, 3) -> RGB directly (each row in [0, 1]). Recommended for
                        S^2: feed normalized (x, y, z) coordinates of node
                        means - invariant to the SO(3) ambiguity of the
                        ell=1 spherical-harmonic eigenspace.
      - None        -> uniform light color.

    Order matches `graph['nodes'].keys()` iteration order.
    """
    G = nx.Graph()
    G.add_nodes_from(graph['nodes'].keys())
    for nid, nbrs in graph.get('links', {}).items():
        for nb in nbrs:
            G.add_edge(nid, nb)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    pos = nx.spring_layout(G, seed=layout_seed)

    sizes = [20 + 4 * len(graph['nodes'][nid]) for nid in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.7)

    if node_color is None:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color='lightsteelblue', node_size=sizes,
            edgecolors='k', linewidths=0.3,
        )
    else:
        node_color = np.asarray(node_color)
        if node_color.ndim == 2 and node_color.shape[1] == 3:
            colors = np.clip(node_color, 0.0, 1.0)
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=list(colors), node_size=sizes,
                edgecolors='k', linewidths=0.3,
            )
        else:
            valid = ~np.isnan(node_color)
            color_arr = np.where(valid, node_color, np.nanmean(node_color) if valid.any() else 0.0)
            nodes_pc = nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=color_arr, node_size=sizes,
                cmap=cmap, edgecolors='k', linewidths=0.3,
                vmin=vmin, vmax=vmax,
            )
            if colorbar and fig is not None:
                fig.colorbar(nodes_pc, ax=ax, fraction=0.04, pad=0.02)

    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


# ─── Spectrum vs theory ────────────────────────────────────────────────────


def plot_spectrum_vs_theory(
    emp_eigs: np.ndarray,
    theory_ratios: Iterable[float],
    title: str = '',
    K: int = 20,
) -> plt.Figure:
    """Empirical eigenvalue ratios vs. closed-form Laplace-Beltrami ratios.

    Plots the emp / emp_1 sequence alongside the theoretical sequence.
    Both should agree up to noise on a log-y axis.
    """
    emp = np.sort(np.asarray(emp_eigs, dtype=float))[:K]
    theory = np.asarray(list(theory_ratios)[:K], dtype=float)
    # Use a real numerical-zero threshold: ARPACK can return ~1e-16 for the
    # zero eigenvalue rather than exact 0, which `emp > 0` won't filter.
    nonzero_emp = emp[emp > 1e-10]
    if len(nonzero_emp) == 0:
        raise ValueError("All empirical eigenvalues are zero.")
    emp_ratio = emp / nonzero_emp[0]
    theory_norm = theory.copy()
    if theory_norm[1:].size > 0 and theory_norm[1] > 0:
        theory_norm = theory_norm / theory_norm[1]

    fig, ax = plt.subplots(figsize=(6, 4))
    emp_idx = np.arange(len(emp_ratio))
    theory_idx = np.arange(len(theory_norm))
    ax.plot(emp_idx, np.maximum(emp_ratio, 1e-10), 'o-', label='empirical', color='steelblue')
    ax.plot(theory_idx, np.maximum(theory_norm, 1e-10), 's--',
            label='theory (Laplace-Beltrami)', color='firebrick')
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index $i$')
    ax.set_ylabel(r'$\lambda_i / \lambda_1$')
    ax.set_title(title or 'Empirical vs. closed-form spectrum')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─── TopK phase transition figure ──────────────────────────────────────────


def plot_topk_phase_transition(
    K_values: list[int],
    recovery_metric: list[float],
    expected_threshold: int | None = None,
    metric_name: str = 'log-ratio error',
    title: str = '',
) -> plt.Figure:
    """Plot a recovery score (e.g. log-ratio error or correct-Betti fraction)
    vs. TopK k. If `expected_threshold` is given (e.g. d+1 per H1), draw a
    vertical line marking the predicted phase transition.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_values, recovery_metric, 'o-', color='steelblue')
    if expected_threshold is not None:
        ax.axvline(expected_threshold, ls='--', color='firebrick',
                   label=f'predicted transition K={expected_threshold}')
        ax.legend()
    ax.set_xlabel('TopK $k$')
    ax.set_ylabel(metric_name)
    ax.set_title(title or 'TopK phase-transition sweep')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
