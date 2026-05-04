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
        'helix': 'Open Helix',
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


# ─── PCA comparison: true samples vs SAE post-activations ──────────────────


_LABEL_MAP = {
    'circle': 'Circle', 'torus': 'Torus', 'sphere': 'Sphere',
    'helix': 'Open Helix', 'figure_eight': 'Figure 8',
    'two_circles': 'Two Circles',
}

# Per-topology, per-coord cyclic flag. Cyclic coords get a cyclic colormap
# (twilight) so the wrap-around isn't visualised as a discontinuity; linear
# coords get viridis.
_CYCLIC_BY_TOPOLOGY = {
    'circle': [True],
    'helix': [False],
    'torus': [True, True],          # both angles wrap
    'sphere': [False, True],        # polar in [0, π] linear, azimuthal cyclic
    'figure_eight': [False],
    'two_circles': [False],
}


def _scatter_with_floor(
    ax,
    Xp: np.ndarray,
    color_values: np.ndarray,
    cmap: str,
    *,
    point_size: float = 10.0,
    alpha: float = 0.45,
) -> None:
    """Scatter ``Xp`` (N, 3) above a translucent floor plane with a soft
    shadow projection. Shared between `plot_pca_comparison` and
    `plot_pca_sparsity_grid` so visual style stays consistent.
    """
    zmin = float(Xp[:, 2].min()); zmax = float(Xp[:, 2].max())
    zspan = max(zmax - zmin, 1e-6)
    zfloor = zmin - 0.15 * zspan

    xmin, xmax = float(Xp[:, 0].min()), float(Xp[:, 0].max())
    ymin, ymax = float(Xp[:, 1].min()), float(Xp[:, 1].max())
    xpad = 0.05 * max(xmax - xmin, 1e-6)
    ypad = 0.05 * max(ymax - ymin, 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(xmin - xpad, xmax + xpad, 2),
        np.linspace(ymin - ypad, ymax + ypad, 2),
    )
    zz = np.full_like(xx, zfloor)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.22,
                    linewidth=0, antialiased=True, shade=False, zorder=0)
    # Soft shadow just above the plane.
    ax.scatter(Xp[:, 0], Xp[:, 1],
               np.full(Xp.shape[0], zfloor + 0.005 * zspan),
               c='gray', s=max(point_size * 0.5, 3),
               alpha=0.07, linewidths=0,
               depthshade=False, zorder=1)
    # Floating cloud.
    ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2],
               c=color_values, s=point_size, alpha=alpha,
               cmap=cmap, linewidths=0, depthshade=True, zorder=2)


def _extract_intrinsic_coords(gt: np.ndarray, topology: str) -> np.ndarray:
    """Recover intrinsic parameters from the saved gt embedding.

    DGP classes save gt as a canonical embedding (e.g. (cos θ, sin θ) for
    circle) rather than the parameter itself; for coloring we want the
    intrinsic parameter so a cyclic colormap maps cleanly across the
    manifold without the cos/sin redundancy.
    """
    if topology == 'circle':
        # gt = (cos θ, sin θ)
        return np.arctan2(gt[:, 1], gt[:, 0])[:, None]
    if topology == 'torus':
        # gt = (cos θ, sin θ, cos φ, sin φ)
        theta = np.arctan2(gt[:, 1], gt[:, 0])
        phi = np.arctan2(gt[:, 3], gt[:, 2])
        return np.column_stack([theta, phi])
    if topology == 'sphere':
        # gt = (x, y, z) on S^2
        polar = np.arccos(np.clip(gt[:, 2], -1.0, 1.0))
        azim = np.arctan2(gt[:, 1], gt[:, 0])
        return np.column_stack([polar, azim])
    if topology == 'helix':
        return gt if gt.ndim == 2 else gt[:, None]
    return gt if gt.ndim == 2 else gt[:, None]


def plot_pca_comparison(
    eval_X: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
    topology: str,
) -> plt.Figure:
    """n_coord-row × 2-col 3D PCA scatter: true samples (left) vs SAE post (right).

    Each row is colored by one intrinsic ground-truth coordinate (extracted
    from the saved canonical embedding by `_extract_intrinsic_coords`).
    Each panel includes a gray XY-plane shadow below the cloud for depth cue.
    No colorbar — coloring is for visual continuity only.
    """
    if gt.ndim == 1:
        gt = gt[:, None]
    coords = _extract_intrinsic_coords(gt, topology)
    n_coords = coords.shape[1]
    cyclic_flags = list(_CYCLIC_BY_TOPOLOGY.get(topology, []))
    if len(cyclic_flags) < n_coords:
        cyclic_flags += [False] * (n_coords - len(cyclic_flags))

    pca_in = PCA(n_components=3).fit_transform(eval_X)
    pca_out = PCA(n_components=3).fit_transform(post)

    fig, axes = plt.subplots(n_coords, 2, figsize=(11, 5.0 * n_coords),
                             subplot_kw={'projection': '3d'}, squeeze=False)
    for c in range(n_coords):
        cmap = 'twilight' if cyclic_flags[c] else 'viridis'
        for col, (Xp, label) in enumerate([(pca_in, 'true samples'),
                                            (pca_out, 'post-activations')]):
            ax = axes[c, col]
            _scatter_with_floor(ax, Xp, coords[:, c], cmap,
                                point_size=11, alpha=0.5)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
            coord_lbl = f' · gt[{c}]' if n_coords > 1 else ''
            ax.set_title(f'{label}{coord_lbl}')
    fig.suptitle(
        f'{_LABEL_MAP.get(topology, topology)} — PCA of true samples '
        f'vs SAE post-activations',
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# ─── Sparsity-ordered PCA grid ─────────────────────────────────────────────


def plot_pca_sparsity_grid(
    eval_X: np.ndarray,
    posts: list,
    gt: np.ndarray,
    topology: str,
) -> plt.Figure:
    """One row per intrinsic GT coord, (1 + len(posts)) cols.

    Col 0: 3D PCA of true samples.
    Cols 1..N: 3D PCA of post-activations, ordered L→R by *increasing*
    sparsity (i.e. decreasing mean L0). `posts` is a list of
    (post_array, knob_label, mean_l0) tuples; the caller is responsible for
    sorting it.

    Same coloring + shadow conventions as `plot_pca_comparison`.
    """
    if gt.ndim == 1:
        gt = gt[:, None]
    coords = _extract_intrinsic_coords(gt, topology)
    n_coords = coords.shape[1]
    cyclic_flags = list(_CYCLIC_BY_TOPOLOGY.get(topology, []))
    if len(cyclic_flags) < n_coords:
        cyclic_flags += [False] * (n_coords - len(cyclic_flags))

    n_cols = 1 + len(posts)
    fig, axes = plt.subplots(n_coords, n_cols,
                             figsize=(3.6 * n_cols, 4.0 * n_coords),
                             subplot_kw={'projection': '3d'}, squeeze=False)

    pca_in = PCA(n_components=3).fit_transform(eval_X)
    pca_posts = [PCA(n_components=3).fit_transform(p) for p, _, _ in posts]

    for c in range(n_coords):
        cmap = 'twilight' if cyclic_flags[c] else 'viridis'
        for col_idx in range(n_cols):
            ax = axes[c, col_idx]
            if col_idx == 0:
                Xp = pca_in
                title = 'true samples'
            else:
                Xp = pca_posts[col_idx - 1]
                _, lbl, l0 = posts[col_idx - 1]
                title = f'{lbl}  (L0={l0:.1f})'
            _scatter_with_floor(ax, Xp, coords[:, c], cmap,
                                point_size=8, alpha=0.45)
            ax.set_xlabel('PC1', labelpad=-2)
            ax.set_ylabel('PC2', labelpad=-2)
            ax.set_zlabel('PC3', labelpad=-2)
            ax.tick_params(axis='both', which='major', labelsize=7)
            coord_lbl = f' · gt[{c}]' if n_coords > 1 else ''
            ax.set_title(f'{title}{coord_lbl}', fontsize=9)
    fig.suptitle(
        f'{_LABEL_MAP.get(topology, topology)} — PCA along sparsity sweep '
        f'(left → right = increasing sparsity)',
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ─── Feature tuning curves (1-D manifolds) ────────────────────────────────


_PARAM_LABEL = {
    'circle': r'$\theta$ (radians)',
    'helix':  'arc length  $s$',
}


def plot_feature_tuning_curves(
    post: np.ndarray,
    gt: np.ndarray,
    topology: str,
    *,
    top_n: int = 8,
    smooth_frac: float = 0.005,
) -> plt.Figure:
    """For a 1-D manifold (circle, helix): plot the top-N features' activation
    traces as a function of the intrinsic GT parameter.

    Each feature is rendered as a faint raw scatter + a bold moving-average
    line. Smoothing window is ``smooth_frac * N`` samples.
    """
    if topology not in ('circle', 'helix'):
        raise ValueError(
            f"tuning curves only defined for 1-D manifolds; got {topology}")
    coords = _extract_intrinsic_coords(gt, topology)
    param = coords[:, 0]
    order = np.argsort(param)
    param_s = param[order]
    post_s = post[order]

    feat_sum = post.sum(axis=0)
    top_features = np.argsort(feat_sum)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.colormaps.get_cmap('tab10')
    win = max(int(smooth_frac * len(param_s)), 1)
    kernel = np.ones(win) / win
    for i, fid in enumerate(top_features):
        y = post_s[:, fid]
        color = cmap(i % 10)
        # Faint raw trace.
        ax.plot(param_s, y, lw=0.4, alpha=0.18, color=color)
        # Bold smoothed trace.
        y_smooth = np.convolve(y, kernel, mode='same')
        ax.plot(param_s, y_smooth, lw=2.2, color=color,
                label=f'feat #{int(fid)}')

    ax.set_xlabel(_PARAM_LABEL.get(topology, 'GT parameter'))
    ax.set_ylabel('Feature activation')
    ax.set_title(
        f'{_LABEL_MAP.get(topology, topology)} — top {top_n} feature '
        f'tuning curves'
    )
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─── Receptive field overlay (all manifolds) ───────────────────────────────


def plot_receptive_fields(
    eval_X: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
    topology: str,
    *,
    top_n: int = 6,
) -> plt.Figure:
    """For each of the top-N features, render the true manifold (3-D PCA)
    colored by that feature's activation — its receptive field on the
    manifold. Shared 3-D layout across panels so feature-to-feature
    comparison is visual.

    The ``gt`` argument is unused for coloring (we color by activation, not
    GT), but is accepted for API symmetry with the other plot functions.
    """
    del gt  # unused; kept for API symmetry

    feat_sum = post.sum(axis=0)
    top_features = np.argsort(feat_sum)[-top_n:][::-1]

    pca_in = PCA(n_components=3).fit_transform(eval_X)

    n_cols = min(top_n, 3)
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.4 * n_rows),
                             subplot_kw={'projection': '3d'}, squeeze=False)
    for i, fid in enumerate(top_features):
        ax = axes[i // n_cols, i % n_cols]
        activation = post[:, fid]
        _scatter_with_floor(ax, pca_in, activation, 'magma',
                            point_size=8, alpha=0.55)
        ax.set_xlabel('PC1', labelpad=-2)
        ax.set_ylabel('PC2', labelpad=-2)
        ax.set_zlabel('PC3', labelpad=-2)
        ax.tick_params(axis='both', which='major', labelsize=7)
        max_act = float(activation.max())
        n_active = int((activation > 0).sum())
        ax.set_title(
            f'feat #{int(fid)}  (max={max_act:.2f}, '
            f'{n_active/len(activation):.1%} active)',
            fontsize=9,
        )
    # Blank any unused panels.
    for j in range(top_n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    fig.suptitle(
        f'{_LABEL_MAP.get(topology, topology)} — top {top_n} feature '
        f'receptive fields',
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ─── Spectral error vs sparsity ────────────────────────────────────────────


def plot_spectral_error_vs_sparsity(
    finite_rows: list,
    diverged_rows: list,
    topology: str,
) -> plt.Figure:
    """One panel per topology: spectral log-ratio error E vs mean L0.

    Broken y-axis: a thin upper panel renders cells whose post-activation
    Coifman–Lafon spectrum diverged (E = ±inf — happens when the
    diffusion graph fragments into many components, e.g. low-K TopK) at a
    single "∞" tick; a large lower panel renders the finite values with
    seed errorbars. Diagonal break marks at the join.

    Parameters
    ----------
    finite_rows
        Per-cell aggregates with all-finite seeds: ``arch, mean_l0,
        mean_l0_std, E_mean, E_std, n_seeds``.
    diverged_rows
        Per-cell aggregates from seeds that returned non-finite E:
        ``arch, mean_l0, mean_l0_std, n_seeds_diverged``.
    """
    arch_style = {
        'relu_l1': dict(color='magenta', label='ReLU+L1', marker='s'),
        'topk':    dict(color='black',   label='TopK',    marker='s'),
    }

    has_diverged = bool(diverged_rows)
    if has_diverged:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [1, 6], 'hspace': 0.05},
            figsize=(7.8, 5.4),
        )
    else:
        fig, ax_bot = plt.subplots(figsize=(7.6, 4.8))
        ax_top = None

    legend_handles = {}

    # Bottom panel — finite values: mean line + shaded ±σ error band.
    for arch, style in arch_style.items():
        sub = sorted([r for r in finite_rows if r['arch'] == arch],
                     key=lambda r: r['mean_l0'])
        if not sub:
            continue
        x = np.array([r['mean_l0'] for r in sub])
        y = np.array([r['E_mean'] for r in sub])
        yerr = np.array([r['E_std'] for r in sub])
        # Clip lower band so log-y doesn't blow up on near-zero values.
        y_lo = np.maximum(y - yerr, np.finfo(float).tiny)
        y_hi = y + yerr
        ax_bot.fill_between(x, y_lo, y_hi, color=style['color'],
                            alpha=0.20, linewidth=0)
        line, = ax_bot.plot(x, y, style['marker'] + '-',
                            color=style['color'], lw=2, ms=6,
                            label=style['label'])
        legend_handles[arch] = line

    # Top panel — diverged cells at "∞".
    if ax_top is not None:
        for arch, style in arch_style.items():
            sub = sorted([r for r in diverged_rows if r['arch'] == arch],
                         key=lambda r: r['mean_l0'])
            if not sub:
                continue
            x = np.array([r['mean_l0'] for r in sub])
            y = np.ones_like(x)
            line, = ax_top.plot(
                x, y, 'x',
                color=style['color'], ms=10, mew=2.2,
                label=style['label'] if arch not in legend_handles else None,
            )
            if arch not in legend_handles:
                legend_handles[arch] = line

        ax_top.set_yticks([1.0])
        ax_top.set_yticklabels([r'$\infty$'], fontsize=13)
        ax_top.set_ylim(0.55, 1.45)
        ax_top.axhspan(0.55, 1.45, color='red', alpha=0.12, zorder=0)
        ax_top.text(
            0.5, 0.93,
            'topology fractured (metric undefined)',
            transform=ax_top.transAxes, ha='center', va='top',
            fontsize=10, color='darkred', fontstyle='italic',
        )
        # Hide spines between subplots and place break marks.
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', which='both', bottom=False,
                           labelbottom=False)
        d = 0.5
        break_kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                            linestyle='none', color='k', mec='k', mew=1,
                            clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes,
                    **break_kwargs)
        ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes,
                    **break_kwargs)
        ax_top.grid(axis='x', alpha=0.3)

    ax_bot.set_xlabel('Mean L0 (per-sample active features)')
    ax_bot.set_ylabel(r'Spectral log-ratio error  $\mathcal{E}$')
    ax_bot.set_xscale('log')
    ax_bot.set_yscale('log')
    ax_bot.grid(alpha=0.3, which='both')
    title = (f'{_LABEL_MAP.get(topology, topology)} — '
             f'spectral error vs sparsity')
    if ax_top is not None:
        ax_top.set_title(title)
    else:
        ax_bot.set_title(title)
    # Combine legend handles from both panels (top-panel artists are not
    # otherwise picked up by ax_bot.legend()).
    handles, labels = ax_bot.get_legend_handles_labels()
    if ax_top is not None:
        h2, l2 = ax_top.get_legend_handles_labels()
        seen = set(labels)
        for h, l in zip(h2, l2):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    if handles:
        ax_bot.legend(handles, labels, loc='best')
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
