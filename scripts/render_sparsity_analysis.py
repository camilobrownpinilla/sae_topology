"""Sparsity-axis analysis: per-(topo, arch, seed) PCA sweep grids and
per-topology spectral-error-vs-sparsity line plots.

Outputs
-------
``<out>/sparsity_grids/<topology>_<arch>_seed<s>.png``
    Sparsity-ordered 3D PCA grid: GT first, then post-activations sorted L→R
    by increasing sparsity (decreasing mean L0).

``<out>/spectral_error_vs_sparsity/<topology>.png``
    Spectral log-ratio error vs mean L0 with seed errorbars; one line per
    arch (ReLU+L1 and TopK).

Usage
-----
    python -m scripts.render_sparsity_analysis \\
        --results_root results/sweep_v1 \\
        --out figures/sweep_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_topology.analysis.plot import (
    plot_pca_sparsity_grid,
    plot_spectral_error_vs_sparsity,
)


N_PLOT = 20_000


def _load_cell(run_dir: Path) -> dict | None:
    rp = run_dir / 'report.json'
    sp = run_dir / 'samples.npz'
    if not (rp.exists() and sp.exists()):
        return None
    with open(rp) as f:
        report = json.load(f)
    s1 = report.get('stage1_result')
    if s1 is None:
        return None
    if 'final_mean_l0' not in s1 or 'spectral_log_ratio_error' not in s1:
        return None
    return {
        'run_dir': run_dir,
        'topology': s1['topology'],
        'arch': s1['arch'],
        'seed': int(s1['seed']),
        'mean_l0': float(s1['final_mean_l0']),
        'spectral_E': float(s1['spectral_log_ratio_error']),
    }


def _knob_str(run_dir: Path) -> str:
    """e.g. 'l1_1e-03' or 'k04' from a run_dir name like 'l1_1e-03_seed0'."""
    return run_dir.name.rsplit('_seed', 1)[0]


def _render_sparsity_grids(cells: list, out_dir: Path) -> int:
    """One grid per (topology, arch, seed)."""
    grouped = defaultdict(list)
    for c in cells:
        grouped[(c['topology'], c['arch'], c['seed'])].append(c)
    n_written = 0
    for (topo, arch, seed), members in sorted(grouped.items()):
        # Sort by mean_l0 descending so leftmost post is the LEAST sparse.
        members_sorted = sorted(members, key=lambda c: c['mean_l0'],
                                reverse=True)
        first = members_sorted[0]
        with np.load(first['run_dir'] / 'samples.npz') as sa:
            eval_X = np.asarray(sa['eval_X'])
            gt = np.asarray(sa['gt'])
        n = eval_X.shape[0]
        rng = np.random.default_rng(0)
        if n > N_PLOT:
            idx = rng.choice(n, N_PLOT, replace=False)
            eval_X_s, gt_s = eval_X[idx], gt[idx]
        else:
            idx = np.arange(n)
            eval_X_s, gt_s = eval_X, gt
        # NOTE: each cell is trained on its OWN eval_X (different seed for
        # the eval-batch sampler), but for visual comparison we use the
        # first cell's eval_X. The post-activation arrays come from each
        # individual run's own eval_X, so we cannot index into them with
        # `idx`. Reload+subsample each independently.
        posts = []
        for c in members_sorted:
            with np.load(c['run_dir'] / 'samples.npz') as sa:
                post = np.asarray(sa['post'])
            n_p = post.shape[0]
            if n_p > N_PLOT:
                idx_p = np.random.default_rng(0).choice(n_p, N_PLOT,
                                                         replace=False)
                post = post[idx_p]
            posts.append((post, _knob_str(c['run_dir']), c['mean_l0']))
        fig = plot_pca_sparsity_grid(eval_X_s, posts, gt_s, topo)
        out = out_dir / f'{topo}_{arch}_seed{seed}.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  ok: {out}')
        n_written += 1
    return n_written


def _render_spectral_error(cells: list, out_dir: Path) -> int:
    """One plot per topology, two lines (arch) with seed errorbars.

    Splits seeds per (arch, knob) cell into finite / diverged so the plot
    function can render diverged seeds at the broken-axis "∞" tick.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for c in cells:
        knob = _knob_str(c['run_dir'])
        grouped[c['topology']][(c['arch'], knob)].append(c)
    n_written = 0
    for topo, by_cell in sorted(grouped.items()):
        finite_rows: list = []
        diverged_rows: list = []
        for (arch, knob), members in by_cell.items():
            finite = [m for m in members if np.isfinite(m['spectral_E'])]
            diverged = [m for m in members if not np.isfinite(m['spectral_E'])]
            if finite:
                l0s = np.array([m['mean_l0'] for m in finite])
                es = np.array([m['spectral_E'] for m in finite])
                finite_rows.append({
                    'arch': arch, 'knob': knob,
                    'mean_l0': float(l0s.mean()),
                    'mean_l0_std': float(l0s.std(ddof=0)),
                    'E_mean': float(es.mean()),
                    'E_std': float(es.std(ddof=0)),
                    'n_seeds': len(finite),
                })
            if diverged:
                l0s = np.array([m['mean_l0'] for m in diverged])
                diverged_rows.append({
                    'arch': arch, 'knob': knob,
                    'mean_l0': float(l0s.mean()),
                    'mean_l0_std': float(l0s.std(ddof=0)),
                    'n_seeds_diverged': len(diverged),
                })
        fig = plot_spectral_error_vs_sparsity(finite_rows, diverged_rows, topo)
        out = out_dir / f'{topo}.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        n_finite = sum(r['n_seeds'] for r in finite_rows)
        n_div = sum(r['n_seeds_diverged'] for r in diverged_rows)
        print(f'  ok: {out}  ({n_finite} finite seeds, {n_div} diverged)')
        n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_root = Path(args.out)
    grid_dir = out_root / 'sparsity_grids'
    spec_dir = out_root / 'spectral_error_vs_sparsity'
    grid_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for d in sorted({p.parent for p in results_root.rglob('report.json')}):
        info = _load_cell(d)
        if info is not None:
            cells.append(info)
    print(f"loaded {len(cells)} cells with valid metrics")

    print("[render_sparsity_analysis] sparsity-ordered PCA grids:")
    n_g = _render_sparsity_grids(cells, grid_dir)
    print(f"[render_sparsity_analysis] spectral error vs sparsity:")
    n_s = _render_spectral_error(cells, spec_dir)
    print(f"[render_sparsity_analysis] done. grids={n_g}  spec_plots={n_s}")


if __name__ == '__main__':
    main()
