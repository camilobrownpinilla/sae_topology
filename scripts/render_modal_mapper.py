"""Post-hoc Mapper graph renderer for already-completed Stage 1 runs.

For each run directory under ``--results_root`` (recursively, identified by
the presence of ``report.json`` + ``samples.npz`` + ``spectrum.npz`` +
``mapper_sweep.json``), recompute Mapper graphs for the stable / modal
Betti cells and write them to ``figures/stable_region_renderings/`` or
``figures/modal_renderings/``.

This mirrors the in-pipeline rendering block in ``stage1_bundle.py`` but
reads everything from disk so existing FAIL runs get visualised without
retraining. Skips runs that already have a populated rendering directory
unless ``--force`` is supplied.

Usage:

    python -m scripts.render_modal_mapper \\
        --results_root results/sweep_v1 [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path so `sae_topology` imports work whether the
# script is invoked as a module or a file.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_topology.mapper import (
    laplacian_eigenvector_filter,
    run_mapper_once,
    stable_betti,
)
from sae_topology.experiments.stage0_plots import (
    mapper_graph_renderings,
    stable_configs_from_correct_region,
)
from sae_topology.experiments.stage0_validate import (
    EXPECTED_BETTI,
    FILTER_K_BY_TOPOLOGY,
)


def _deserialise_sweep(records: list[dict]) -> dict:
    """Invert ``stage0_artifacts._serialise_sweep`` back to a (ni, ov) dict."""
    sweep: dict = {}
    for rec in records:
        ni = int(rec['n_intervals'])
        ov = float(rec['overlap'])
        entry = {k: v for k, v in rec.items() if k not in ('n_intervals', 'overlap')}
        sweep[(ni, ov)] = entry
    return sweep


def _has_existing_renderings(figures_dir: Path) -> bool:
    for sub in ('stable_region_renderings', 'modal_renderings'):
        d = figures_dir / sub
        if d.is_dir() and any(d.glob('*.png')):
            return True
    return False


def render_one(run_dir: Path, *, force: bool = False) -> str:
    """Render Mapper graphs for one run dir. Returns a status string.

    Single-BLAS-thread context inside the worker so loky workers don't
    over-subscribe the node when --n_jobs > 1.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _render_one_inner(run_dir, force=force)


def _render_one_inner(run_dir: Path, *, force: bool = False) -> str:
    report_p = run_dir / 'report.json'
    samples_p = run_dir / 'samples.npz'
    spectrum_p = run_dir / 'spectrum.npz'
    sweep_p = run_dir / 'mapper_sweep.json'
    if not (report_p.exists() and samples_p.exists()
            and spectrum_p.exists() and sweep_p.exists()):
        return f"skip (missing artifacts): {run_dir}"

    figures_dir = run_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    if not force and _has_existing_renderings(figures_dir):
        return f"skip (already rendered): {run_dir}"

    with open(report_p) as f:
        report = json.load(f)
    topology = report['topology']
    meta = report.get('_meta', {})
    k_filter = int(meta.get('k_filter', FILTER_K_BY_TOPOLOGY.get(topology, 3)))
    threshold = float(meta['distance_threshold'])

    with np.load(samples_p) as sa:
        post = sa['post']
        gt = sa['gt'] if 'gt' in sa.files else None
        if gt is not None and gt.size == 0:
            gt = None
    with np.load(spectrum_p) as sp:
        eigenvectors = sp['eigenvectors']

    with open(sweep_p) as f:
        sweep_blob = json.load(f)
    sweep = _deserialise_sweep(sweep_blob['laplacian'])

    expected = EXPECTED_BETTI.get(topology, (1, 0))
    stable_cells = stable_configs_from_correct_region(sweep, expected)
    if stable_cells:
        cells_to_render, render_subdir = stable_cells, 'stable_region_renderings'
    else:
        modal_b0, modal_b1, _ = stable_betti(sweep)
        cells_to_render = stable_configs_from_correct_region(
            sweep, (modal_b0, modal_b1),
        )
        render_subdir = 'modal_renderings'

    if not cells_to_render:
        return f"skip (no cells matched modal Betti): {run_dir}"

    lens = laplacian_eigenvector_filter(post, k=k_filter, eigenvectors=eigenvectors)
    graph_by_cell: dict[tuple[int, float], dict] = {}
    for (ni, ov) in cells_to_render:
        graph_by_cell[(int(ni), float(ov))] = run_mapper_once(
            post, lens, int(ni), float(ov), distance_threshold=threshold,
        )
    out_dir = figures_dir / render_subdir
    mapper_graph_renderings(
        graph_by_config=graph_by_cell,
        gt=(np.asarray(gt) if gt is not None else None),
        topology=topology,
        stable_configs=cells_to_render,
        out_dir=out_dir,
    )
    return (f"rendered {len(cells_to_render)} {render_subdir}: "
            f"{run_dir} -> {out_dir.relative_to(run_dir)}")


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob('report.json'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results_root', required=True,
                    help='Root directory under which to find Stage 1 runs.')
    ap.add_argument('--force', action='store_true',
                    help='Re-render even if figures already exist.')
    ap.add_argument('--n_jobs', type=int, default=1,
                    help='Number of loky workers (1 run per worker).')
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    run_dirs = find_run_dirs(root)
    print(f"[render-modal] found {len(run_dirs)} runs under {root} "
          f"(n_jobs={args.n_jobs})", flush=True)

    def _safe_render(rd: Path) -> str:
        try:
            return render_one(rd, force=args.force)
        except Exception as e:
            return f"ERROR: {rd}: {type(e).__name__}: {e}"

    if args.n_jobs == 1 or len(run_dirs) <= 1:
        results = []
        for rd in run_dirs:
            msg = _safe_render(rd)
            print(msg, flush=True)
            results.append(msg)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=10)(
            delayed(_safe_render)(rd) for rd in run_dirs
        )
        for msg in results:
            print(msg, flush=True)

    n_rendered = sum(1 for m in results if m.startswith('rendered'))
    n_skipped = sum(1 for m in results if m.startswith('skip'))
    n_error = sum(1 for m in results if m.startswith('ERROR'))
    print(f"[render-modal] done: {n_rendered} rendered, "
          f"{n_skipped} skipped, {n_error} errors", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
