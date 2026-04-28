"""Measure wall-clock speedup of the Stage 0 flat-parallel pipeline at
varying `n_jobs`.

For each n_jobs in `--jobs` (default: 1, 4, 8, 16, 32, 56), runs the full
Stage 0 auto-tune for the requested topologies and prints:

    n_jobs=N : wall=Xs (speedup vs n_jobs=1 = Yx)

Use `--n_samples N` to override the per-topology sample size (useful for
benchmarking at the user's intended scale, e.g. N=100k for 2-manifolds).

Run:  python -m sae_topology.experiments.measure_parallel_speedup
      python -m sae_topology.experiments.measure_parallel_speedup \
          --topologies circle torus --jobs 1 8 32 --n_samples 50000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sae_topology.experiments.stage0_validate import run_stage0


def main():
    parser = argparse.ArgumentParser(
        description="Measure Stage 0 wall-clock at varying n_jobs."
    )
    parser.add_argument('--topologies', nargs='+',
                        default=['circle', 'torus', 'sphere', 'figure_eight'])
    parser.add_argument('--jobs', nargs='+', type=int,
                        default=[1, 4, 8, 16, 32, 56])
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Override per-topology N (default: per-topology defaults).')
    parser.add_argument('--ambient_d', type=int, default=64)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--quick', action='store_true',
                        help='Halve sample sizes (ignored if --n_samples is set).')
    parser.add_argument('--out_dir', default='results/parallel_speedup',
                        help='Where to write per-n_jobs baseline.yaml + json.')
    parser.add_argument('--check_identical', action='store_true',
                        help='Diff each n_jobs>1 baseline against n_jobs=1; '
                             'exit nonzero if any differ.')
    args = parser.parse_args()

    n_samples_override = None
    if args.n_samples is not None:
        n_samples_override = {topo: args.n_samples for topo in args.topologies}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[int, float] = {}
    yaml_paths: dict[int, Path] = {}

    for nj in args.jobs:
        out_yaml = out_dir / f'baseline_n{nj}.yaml'
        out_json = out_dir / f'baseline_n{nj}.json'
        yaml_paths[nj] = out_yaml
        print(f"\n{'=' * 60}")
        print(f"n_jobs={nj}: starting Stage 0 over {args.topologies}")
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()
        run_stage0(
            topologies=args.topologies,
            out_yaml=out_yaml,
            out_json=out_json,
            quick=args.quick,
            n_jobs=nj,
            ambient_d=args.ambient_d,
            sigma=args.sigma,
            n_samples_override=n_samples_override,
        )
        wall = time.time() - t0
        timings[nj] = wall
        print(f"\n--> n_jobs={nj}: wall={wall:.2f}s")

    print(f"\n\n{'=' * 60}")
    print("Speedup summary")
    print(f"{'=' * 60}")
    if 1 in timings:
        baseline = timings[1]
        for nj in args.jobs:
            ratio = baseline / timings[nj]
            print(f"  n_jobs={nj:3d}: wall={timings[nj]:8.2f}s  (speedup vs n_jobs=1: {ratio:5.2f}x)")
    else:
        for nj in args.jobs:
            print(f"  n_jobs={nj:3d}: wall={timings[nj]:8.2f}s")

    if args.check_identical and 1 in timings:
        print(f"\n--check_identical: diffing each baseline against n_jobs=1 ...")
        ref = yaml_paths[1].read_bytes()
        any_diff = False
        for nj in args.jobs:
            if nj == 1:
                continue
            this = yaml_paths[nj].read_bytes()
            if this != ref:
                print(f"  n_jobs={nj}: DIFFERS from n_jobs=1 baseline (FAIL)")
                any_diff = True
            else:
                print(f"  n_jobs={nj}: identical")
        if any_diff:
            raise SystemExit(1)
        print("  all baselines identical.")


if __name__ == '__main__':
    main()
