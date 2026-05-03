"""Assemble the headline poster figures.

Two outputs:

1. ``figures/poster/tuning_curves_2x2.png`` — 2 rows (circle, helix) × 2
   cols (ReLU+L1, TopK). Each panel = top-N feature tuning curves.

2. ``figures/poster/receptive_fields_4x2.png`` — 4 rows (circle, helix,
   torus, sphere) × 2 cols (ReLU+L1, TopK). Each panel = one
   representative feature's receptive field on the true manifold's 3D PCA.

Pulls representative cells from the existing sweep:
    relu_l1 → l1_1e-03_seed0
    topk    → k04_seed0
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_topology.analysis.plot import (
    _LABEL_MAP, _CYCLIC_BY_TOPOLOGY, _PARAM_LABEL,
    _extract_intrinsic_coords, _scatter_with_floor,
)


N_PLOT = 20_000
RESULTS = REPO_ROOT / 'results' / 'sweep_v1'
OUT_DIR = REPO_ROOT / 'figures' / 'poster'

# Representative cells.
RELU_KNOB = 'l1_1e-03'
# Try these in order until a cell with samples.npz exists. K=4 gives clean
# multi-tile visuals where available; K=3 / K=2 fall back as more sphere /
# torus cells are still running.
TOPK_KNOB_PREFERENCES = ['k04', 'k03', 'k02']
SEED = 0

_ARCH_COL = {'relu_l1': 0, 'topk': 1}
_ARCH_LABEL = {'relu_l1': 'ReLU + L1', 'topk': 'TopK'}


def _resolve_topk_knob(topology: str) -> str:
    for k in TOPK_KNOB_PREFERENCES:
        if (RESULTS / topology / 'topk' / f'{k}_seed{SEED}' /
                'samples.npz').exists():
            return k
    raise FileNotFoundError(
        f'no TopK seed{SEED} cell available for {topology}; '
        f'tried {TOPK_KNOB_PREFERENCES}'
    )


def _load_cell(topology: str, arch: str) -> dict:
    if arch == 'relu_l1':
        knob = RELU_KNOB
    else:
        knob = _resolve_topk_knob(topology)
    run_dir = RESULTS / topology / arch / f'{knob}_seed{SEED}'
    with np.load(run_dir / 'samples.npz') as sa:
        eval_X = np.asarray(sa['eval_X'])
        post = np.asarray(sa['post'])
        gt = np.asarray(sa['gt'])
    n = eval_X.shape[0]
    if n > N_PLOT:
        idx = np.random.default_rng(0).choice(n, N_PLOT, replace=False)
        eval_X, post, gt = eval_X[idx], post[idx], gt[idx]
    return {'eval_X': eval_X, 'post': post, 'gt': gt,
            'run_dir': run_dir, 'topology': topology, 'arch': arch}


def _plot_tuning_curves_on_ax(
    ax,
    post: np.ndarray,
    gt: np.ndarray,
    topology: str,
    *,
    top_n: int = 8,
    smooth_frac: float = 0.005,
) -> None:
    coords = _extract_intrinsic_coords(gt, topology)
    param = coords[:, 0]
    order = np.argsort(param)
    param_s = param[order]
    post_s = post[order]
    feat_sum = post.sum(axis=0)
    top_features = np.argsort(feat_sum)[-top_n:][::-1]

    cmap = plt.colormaps.get_cmap('tab10')
    win = max(int(smooth_frac * len(param_s)), 1)
    kernel = np.ones(win) / win
    for i, fid in enumerate(top_features):
        y = post_s[:, fid]
        c = cmap(i % 10)
        ax.plot(param_s, y, lw=0.4, alpha=0.18, color=c)
        y_smooth = np.convolve(y, kernel, mode='same')
        ax.plot(param_s, y_smooth, lw=2.0, color=c)
    ax.set_xlabel(_PARAM_LABEL.get(topology, 'GT parameter'))
    ax.set_ylabel('Feature activation')
    ax.grid(alpha=0.3)


def _plot_receptive_field_on_ax(
    ax,
    eval_X: np.ndarray,
    post: np.ndarray,
) -> int:
    """Render the top-1 feature's receptive field on the true-manifold 3D PCA.
    Returns the chosen feature ID for downstream labelling.
    """
    feat_sum = post.sum(axis=0)
    fid = int(np.argsort(feat_sum)[-1])
    pca_in = PCA(n_components=3).fit_transform(eval_X)
    activation = post[:, fid]
    _scatter_with_floor(ax, pca_in, activation, 'magma',
                        point_size=10, alpha=0.55)
    ax.set_xlabel('PC1', labelpad=-2)
    ax.set_ylabel('PC2', labelpad=-2)
    ax.set_zlabel('PC3', labelpad=-2)
    ax.tick_params(axis='both', which='major', labelsize=7)
    return fid


def _knob_label(topology: str, arch: str) -> str:
    return RELU_KNOB if arch == 'relu_l1' else _resolve_topk_knob(topology)


def render_tuning_curves_2x2() -> Path:
    rows = ['circle', 'helix']
    cols = ['relu_l1', 'topk']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for r, topo in enumerate(rows):
        for c, arch in enumerate(cols):
            cell = _load_cell(topo, arch)
            ax = axes[r, c]
            _plot_tuning_curves_on_ax(ax, cell['post'], cell['gt'], topo)
            ax.set_title(
                f'{_LABEL_MAP[topo]} — {_ARCH_LABEL[arch]} '
                f'({_knob_label(topo, arch)})',
                fontsize=12,
            )
    fig.suptitle(
        'Feature tuning curves: ReLU+L1 dilutes vs TopK tiles',
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / 'tuning_curves_2x2.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')
    return out


def render_receptive_fields_4x2() -> Path:
    rows = ['circle', 'helix', 'torus', 'sphere']
    cols = ['relu_l1', 'topk']
    fig, axes = plt.subplots(4, 2, figsize=(11, 18),
                             subplot_kw={'projection': '3d'})
    for r, topo in enumerate(rows):
        for c, arch in enumerate(cols):
            cell = _load_cell(topo, arch)
            ax = axes[r, c]
            fid = _plot_receptive_field_on_ax(
                ax, cell['eval_X'], cell['post']
            )
            ax.set_title(
                f'{_LABEL_MAP[topo]} — {_ARCH_LABEL[arch]} '
                f'({_knob_label(topo, arch)})\nfeat #{fid}',
                fontsize=11,
            )
    fig.suptitle(
        'Top-1 feature receptive fields on the true manifold',
        fontsize=15, y=1.005,
    )
    fig.tight_layout()
    out = OUT_DIR / 'receptive_fields_4x2.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')
    return out


def main() -> None:
    render_tuning_curves_2x2()
    render_receptive_fields_4x2()


if __name__ == '__main__':
    main()
