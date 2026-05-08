"""Single-cell SAE calibration: train one (manifold, arch, coeff) and dump final L0.

Skips spectral + mapper. Just trains for n_steps and writes a small JSON with
the final mean_l0 and last few entries of the training trace so we can pick
sparsity coefficients that hit a target L0 per manifold.

Usage:
  python -m scripts.calibrate_one \
      --topology helix --arch jumprelu --coeff 1e-2 \
      --n_steps 50000 --seed 0 --out_path calib/helix_jumprelu_1e-02.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sae_topology.dgp.spaces import Circle, Helix, Sphere, Torus
from sae_topology.saes.architectures import build_sae
from sae_topology.saes.config import SAETrainConfig
from sae_topology.saes.trainer import SAETrainer


def make_dgp(topology: str):
    center = np.zeros(64)
    if topology == "circle":
        return Circle(dimension=64, noise_level=0.01, center=center)
    if topology == "sphere":
        return Sphere(dimension=64, noise_level=0.01, center=center)
    if topology == "torus":
        return Torus(dimension=64, noise_level=0.01, center=center)
    if topology == "helix":
        return Helix(dimension=64, noise_level=0.01, center=center,
                     radius=1.0, pitch=0.5, n_turns=4.0)
    raise ValueError(f"unknown topology {topology}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True)
    p.add_argument("--arch", required=True, choices=["relu_l1", "jumprelu"])
    p.add_argument("--coeff", required=True, type=float)
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--d_sae", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_kwargs = dict(
        arch=args.arch, d_in=64, d_sae=args.d_sae,
        lr=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        resample_every=2500, dead_threshold=1e-3, seed=args.seed,
    )
    if args.arch == "relu_l1":
        cfg_kwargs["l1_coeff"] = args.coeff
    elif args.arch == "jumprelu":
        cfg_kwargs["l0_coeff"] = args.coeff

    cfg = SAETrainConfig(**cfg_kwargs)
    dgp = make_dgp(args.topology)
    sae = build_sae(cfg)
    trainer = SAETrainer(sae, cfg, dgp)

    t0 = time.time()
    out = trainer.train()
    elapsed = time.time() - t0

    m = out["metrics"]
    # Average last 10% of logged points to smooth out batch noise
    n_log = len(m["mean_l0"])
    tail = max(1, n_log // 10)
    l0_tail = float(np.mean(m["mean_l0"][-tail:]))
    mse_tail = float(np.mean(m["mse"][-tail:]))
    n_dead_final = int(m["n_dead"][-1])

    record = dict(
        topology=args.topology, arch=args.arch, coeff=args.coeff, seed=args.seed,
        n_steps=args.n_steps, elapsed_s=elapsed,
        mean_l0_final=float(m["mean_l0"][-1]),
        mean_l0_tail=l0_tail,           # mean over last 10% of steps
        mse_tail=mse_tail,
        n_dead_final=n_dead_final,
        steps_logged=m["step"][::max(1, n_log // 20)],   # subsample
        mean_l0_logged=[float(x) for x in m["mean_l0"][::max(1, n_log // 20)]],
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[calib] {args.topology}/{args.arch}/coeff={args.coeff:g} "
          f"L0_tail={l0_tail:.2f} n_dead={n_dead_final} mse_tail={mse_tail:.5f} "
          f"elapsed={elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
