"""Render the τ (topological stability) bar plot.

τ = mapper_laplacian_correct_region.n_correct_total / total_configs
  = fraction of (n_intervals × overlap) Mapper-grid cells whose graph_betti
    matches the ground-truth Betti for the given manifold.

For each (manifold, arch) we select the runs whose seed-mean L0 is closest to
the L0=32 target across both sweep_v1 and sweep_v2. (For TopK this is k32; for
ReLU+L1 the closest knob varies per manifold; for JumpReLU it's t32.)

Usage
-----
    python -m scripts.render_tau_bar --out figures/sweep_consolidated
"""
from __future__ import annotations

import argparse
import json
import os
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

from sae_topology.analysis.plot import plot_tau_bar


TARGET_L0 = 32.0
MANIFOLDS = ['circle', 'torus', 'sphere', 'helix']
ARCHS = ['relu_l1', 'topk', 'jumprelu']
SWEEPS = ['sweep_v1', 'sweep_v2']


def _knob_str(run_dir: Path) -> str:
    return run_dir.name.rsplit('_seed', 1)[0]


def _load_cell(run_dir: Path) -> dict | None:
    rp = run_dir / 'report.json'
    if not rp.exists():
        return None
    with open(rp) as f:
        report = json.load(f)
    s1 = report.get('stage1_result')
    if s1 is None:
        return None
    cr = s1.get('mapper_laplacian_correct_region')
    if cr is None:
        return None
    n_correct = cr.get('n_correct_total')
    n_total   = cr.get('total_configs')
    l0        = s1.get('final_mean_l0')
    if n_correct is None or n_total in (None, 0) or l0 is None:
        return None
    return {
        'run_dir':   run_dir,
        'topology':  s1['topology'],
        'arch':      s1['arch'],
        'seed':      int(s1['seed']),
        'l0':        float(l0),
        'tau':       float(n_correct) / float(n_total),
        'knob':      _knob_str(run_dir),
    }


def _select_closest_to_target(cells: list, target: float = TARGET_L0) -> dict:
    """For each (manifold, arch), pick the knob whose seed-mean L0 is
    closest to `target`. Returns {(manifold, arch): chosen_knob}."""
    by_mak = defaultdict(list)  # (manifold, arch, knob) -> [cells]
    for c in cells:
        by_mak[(c['topology'], c['arch'], c['knob'])].append(c)
    knob_l0 = {k: float(np.mean([c['l0'] for c in v])) for k, v in by_mak.items()}

    chosen = {}
    for manifold in MANIFOLDS:
        for arch in ARCHS:
            keys = [(m, a, kn) for (m, a, kn) in knob_l0 if m == manifold and a == arch]
            if not keys:
                continue
            best = min(keys, key=lambda k: abs(knob_l0[k] - target))
            chosen[(manifold, arch)] = best[2]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results',
                    help='walked recursively for sweep_v1/sweep_v2/* results.')
    ap.add_argument('--out', required=True, help='output dir for tau_bar.png.')
    ap.add_argument('--target_l0', type=float, default=TARGET_L0)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect cells from sweep_v1 + sweep_v2 only (avoid the consolidated
    # symlink tree, which would double-count).
    cells = []
    for sweep in SWEEPS:
        base = results_root / sweep
        if not base.is_dir():
            continue
        for root, _, files in os.walk(base, followlinks=True):
            if 'report.json' not in files:
                continue
            info = _load_cell(Path(root))
            if info is not None:
                cells.append(info)
    print(f'loaded {len(cells)} cells with τ available')

    chosen = _select_closest_to_target(cells, target=args.target_l0)
    print(f'\nSelected knob per (manifold, arch) at L0≈{args.target_l0}:')
    for (m, a), kn in sorted(chosen.items()):
        sub = [c for c in cells if c['topology'] == m and c['arch'] == a and c['knob'] == kn]
        l0_mean = float(np.mean([c['l0'] for c in sub]))
        print(f'  {m:8s}/{a:9s}  knob={kn:12s}  n_seeds={len(sub)}  mean_L0={l0_mean:.2f}')

    rows = []
    for (m, a), kn in chosen.items():
        sub = [c for c in cells if c['topology'] == m and c['arch'] == a and c['knob'] == kn]
        taus = np.array([c['tau'] for c in sub])
        rows.append({
            'manifold': m, 'arch': a, 'knob': kn,
            'tau_mean': float(taus.mean()),
            'tau_std':  float(taus.std(ddof=0)),
            'n_seeds':  len(sub),
        })

    fig = plot_tau_bar(
        rows,
        title_suffix=f'L0 ≈ {int(args.target_l0)}',
    )
    out = out_dir / 'tau_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
