"""Build the sweep_topk_high manifest: 36 per-run YAML configs + manifest.tsv.

Sweep matrix:
    4 topologies (circle, helix, torus, sphere)
    x topk arch only
    x k in {16, 24, 32}                                          (3 values)
    x 3 seeds (0, 1, 2)
    = 36 runs

Motivation: existing TopK runs at K in {1..8} produce shattered post-activation
graphs (spectral_E = inf in most cells). ReLU+L1 cells live in L0 ~ 10..110.
Higher-K TopK (16, 24, 32) should both sit in the same sparsity range as
ReLU+L1 *and* keep the post-activation graph well-connected so the
Coifman-Lafon spectrum is finite.

Task ordering inside the manifest cycles (topology, seed) within each K block,
so a partial completion still covers every manifold:
    indices  0..11  -> K=16, (circle,helix,torus,sphere) x (seed0,seed1,seed2)
    indices 12..23  -> K=24, same pattern
    indices 24..35  -> K=32, same pattern

Run dir layout: results/sweep_v1/<topo>/topk/k<K>_seed<s>/  (unified with
sweep_v1 so the same aggregate.py invocation discovers k16/k24/k32 alongside
the existing k01..k08 cells).
Config layout:  configs/sweep_topk_high/<run_id>.yaml
Manifest:       configs/sweep_topk_high/manifest.tsv

NOTE: configs/sweep_v1/manifest.tsv must NOT be rewritten while a sweep_v1
array job is still running, since sweep_v1.sbatch reads it per-task by
SLURM_ARRAY_TASK_ID. This is why the high-K sweep gets its own configs
namespace (separate manifest, separate sbatch).
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO / "configs" / "sweep_topk_high"
RESULTS_DIR = REPO / "results" / "sweep_v1"      # unified namespace
LOGS_DIR = REPO / "logs" / "sweep_topk_high"

TOPOLOGIES = ["circle", "helix", "torus", "sphere"]
SEEDS = [0, 1, 2]
K_GRID = [16, 24, 32]

# Identical to sweep_v1.
FIXED_SAE = {
    "d_sae": 128,
    "lr": 4.0e-4,
    "n_steps": 50_000,
    "batch_size": 256,
    "resample_every": 2_500,
    "dead_threshold": 1.0e-3,
    "n_eval_samples": 100_000,
    "pca_dim": 4,
}
# Per-topology DGP kwargs. Circle/torus/sphere take only (d, sigma); helix
# also takes (radius, pitch, n_turns) which we keep explicit so silent
# Helix.__init__ default drift can't move the geometry.
DGP_BY_TOPO: dict[str, dict] = {
    "circle": {"d": 64, "sigma": 0.01},
    "torus":  {"d": 64, "sigma": 0.01},
    "sphere": {"d": 64, "sigma": 0.01},
    "helix":  {"d": 64, "sigma": 0.01, "radius": 1.0, "pitch": 0.5, "n_turns": 4.0},
}


def _knob_str(k: int) -> str:
    return f"k{k:02d}"


def _build_cell(topology: str, k: int, seed: int) -> dict:
    knob_str = _knob_str(k)
    run_id = f"{topology}_topk_{knob_str}_seed{seed}"
    config_path = CONFIGS_DIR / f"{run_id}.yaml"
    out_dir = RESULTS_DIR / topology / "topk" / f"{knob_str}_seed{seed}"

    sae = {**FIXED_SAE, "arch": "topk", "seed": seed, "k": int(k)}
    yaml_doc = {
        "topology": topology,
        "dgp": dict(DGP_BY_TOPO[topology]),
        "sae": sae,
    }
    return {
        "run_id": run_id,
        "config_path": config_path,
        "out_dir": out_dir,
        "topology": topology,
        "arch": "topk",
        "sparsity_knob": "k",
        "sparsity_value": int(k),
        "seed": seed,
        "yaml_doc": yaml_doc,
    }


def _enumerate_cells() -> list[dict]:
    """Order: K outer, then seed, then topology.

    With %16 throttle and 12 cells per K block, this means all of K=16 starts
    immediately + 4 of K=24; partial completions still cover every topology.
    """
    cells = []
    for k in K_GRID:
        for seed in SEEDS:
            for topology in TOPOLOGIES:
                cells.append(_build_cell(topology, k, seed))
    return cells


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cells = _enumerate_cells()

    for cell in cells:
        with open(cell["config_path"], "w") as f:
            yaml.safe_dump(cell["yaml_doc"], f, sort_keys=False)

    manifest_path = CONFIGS_DIR / "manifest.tsv"
    header = ["idx", "run_id", "config_path", "out_dir",
              "topology", "arch", "sparsity_knob", "sparsity_value", "seed"]
    with open(manifest_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for idx, cell in enumerate(cells):
            f.write("\t".join([
                str(idx),
                cell["run_id"],
                str(cell["config_path"]),
                str(cell["out_dir"]),
                cell["topology"],
                cell["arch"],
                cell["sparsity_knob"],
                f"{cell['sparsity_value']:g}",
                str(cell["seed"]),
            ]) + "\n")

    by_k: dict[int, int] = {}
    by_topo: dict[str, int] = {}
    for c in cells:
        by_k[c["sparsity_value"]] = by_k.get(c["sparsity_value"], 0) + 1
        by_topo[c["topology"]] = by_topo.get(c["topology"], 0) + 1
    print(f"wrote {len(cells)} configs to {CONFIGS_DIR}")
    print(f"  by K:   {by_k}")
    print(f"  by topo: {by_topo}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
