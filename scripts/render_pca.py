"""Per-run PCA renderer: true samples vs SAE post-activations.

For each run dir under ``--results_root`` (recursively, identified by the
presence of ``samples.npz`` + ``report.json``), produce a 2-row figure:

    row 0 = 2D PCA of eval_X (the true sampled manifold)
    row 1 = 2D PCA of post (SAE post-activations)
    cols  = one per ground-truth coord, each colored by gt[:, c]

Output: ``<run_dir>/figures/pca_comparison.png``. Skip-if-exists.

Mirrors ``scripts/render_modal_mapper.py``. PCA + scatter on a 20k subsample
takes <1 s per cell, so this runs serial on the head node — no SLURM needed.

Usage:
    python -m scripts.render_pca --results_root results/sweep_v1 [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_topology.analysis.plot import plot_pca_comparison


N_PLOT = 20_000  # subsample size for the scatter (PCA is fit on this too)


def render_one(run_dir: Path, *, force: bool = False) -> str:
    samples_p = run_dir / 'samples.npz'
    report_p = run_dir / 'report.json'
    if not (samples_p.exists() and report_p.exists()):
        return f"skip (missing artifacts): {run_dir}"

    figures_dir = run_dir / 'figures'
    out = figures_dir / 'pca_comparison.png'
    if out.exists() and not force:
        return f"skip (already rendered): {run_dir}"
    figures_dir.mkdir(exist_ok=True)

    with open(report_p) as f:
        report = json.load(f)
    topology = report['topology']

    with np.load(samples_p) as sa:
        eval_X = sa['eval_X']
        post = sa['post']
        gt = sa['gt']

    # Subsample with a fixed seed so re-renders are stable.
    n = eval_X.shape[0]
    if n > N_PLOT:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, N_PLOT, replace=False)
        eval_X, post, gt = eval_X[idx], post[idx], gt[idx]

    fig = plot_pca_comparison(eval_X, post, gt, topology)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return f"ok: {out}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = Path(args.results_root)
    run_dirs = sorted({p.parent for p in root.rglob('report.json')})
    print(f"[render_pca] found {len(run_dirs)} run dir(s) under {root}")
    n_ok = n_skip = 0
    for d in run_dirs:
        msg = render_one(d, force=args.force)
        print(msg, flush=True)
        if msg.startswith('ok:'):
            n_ok += 1
        else:
            n_skip += 1
    print(f"[render_pca] done. ok={n_ok} skip={n_skip}")


if __name__ == '__main__':
    main()
