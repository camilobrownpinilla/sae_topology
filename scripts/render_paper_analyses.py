"""Per-run feature tuning curves + receptive field overlays.

Inspired by Bhalla et al. 2026 ("Do SAEs Capture Concept Manifolds?"):
    - Tuning curves (Figure 7-style): top-N feature activations as a
      function of the intrinsic GT parameter. 1-D manifolds only
      (circle, helix).
    - Receptive field overlays (Figure 8/9-style): true manifold
      colored by each top-N feature's activation. All manifolds.

Outputs (per run dir):
    figures/tuning_curves.png       (only for 1-D manifolds)
    figures/receptive_fields.png    (all topologies)

Skip-if-exists. Subsamples to ``N_PLOT`` for speed.

Usage
-----
    python -m scripts.render_paper_analyses \\
        --results_root results/sweep_v1 [--force]
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

from sae_topology.analysis.plot import (
    plot_feature_tuning_curves,
    plot_receptive_fields,
)


N_PLOT = 20_000
ONE_D_MANIFOLDS = {'circle', 'helix'}


def render_one(run_dir: Path, *, force: bool = False) -> str:
    samples_p = run_dir / 'samples.npz'
    report_p = run_dir / 'report.json'
    if not (samples_p.exists() and report_p.exists()):
        return f"skip (missing artifacts): {run_dir}"

    figures_dir = run_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tc_out = figures_dir / 'tuning_curves.png'
    rf_out = figures_dir / 'receptive_fields.png'

    with open(report_p) as f:
        report = json.load(f)
    topology = report['topology']

    will_render_tc = topology in ONE_D_MANIFOLDS
    tc_done = tc_out.exists()
    rf_done = rf_out.exists()
    if not force and rf_done and (not will_render_tc or tc_done):
        return f"skip (already rendered): {run_dir}"

    with np.load(samples_p) as sa:
        eval_X = np.asarray(sa['eval_X'])
        post = np.asarray(sa['post'])
        gt = np.asarray(sa['gt'])

    n = eval_X.shape[0]
    if n > N_PLOT:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, N_PLOT, replace=False)
        eval_X = eval_X[idx]; post = post[idx]; gt = gt[idx]

    msgs = []
    if will_render_tc and (force or not tc_done):
        fig = plot_feature_tuning_curves(post, gt, topology)
        fig.savefig(tc_out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        msgs.append(f'ok: {tc_out}')
    if force or not rf_done:
        fig = plot_receptive_fields(eval_X, post, gt, topology)
        fig.savefig(rf_out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        msgs.append(f'ok: {rf_out}')
    return ' | '.join(msgs) if msgs else f"skip: {run_dir}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = Path(args.results_root)
    run_dirs = sorted({p.parent for p in root.rglob('report.json')})
    print(f"[render_paper_analyses] found {len(run_dirs)} run dir(s)")
    n_ok = n_skip = 0
    for d in run_dirs:
        msg = render_one(d, force=args.force)
        print(msg, flush=True)
        if msg.startswith('ok:') or '| ok:' in msg:
            n_ok += 1
        else:
            n_skip += 1
    print(f"[render_paper_analyses] done. ok={n_ok} skip={n_skip}")


if __name__ == '__main__':
    main()
