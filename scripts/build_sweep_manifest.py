"""Build the sweep_v1 manifest: 144 per-run YAML configs + manifest.tsv.

Sweep matrix:
    3 topologies (circle, torus, sphere)
    x 3 archs:
        relu_l1   l1_coeff in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}    (5 values)
        topk      k        in {1, 2, 3, 4, 6, 8}                (6 values)
        jumprelu  l0_coeff in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}    (5 values)
    x 3 seeds (0, 1, 2)
    = 144 runs

Run dir layout: results/sweep_v1/<topology>/<arch>/<knob_str>_seed<s>/
Config layout:  configs/sweep_v1/<run_id>.yaml
Manifest:       configs/sweep_v1/manifest.tsv
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO / "configs" / "sweep_v1"
RESULTS_DIR = REPO / "results" / "sweep_v1"
LOGS_DIR = REPO / "logs" / "sweep_v1"

TOPOLOGIES = ["circle", "torus", "sphere"]
SEEDS = [0, 1, 2]

L1_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
K_GRID = [1, 2, 3, 4, 6, 8]
# Recalibrated post SAELens-port: knee lives between 1e-3 and 1e-2.
# See /tmp/l0_calibrate.py results — anything >= 3e-2 collapses to all-dead.
L0_GRID = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3]

# Fixed per run (overrides SAETrainConfig defaults).
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
FIXED_DGP = {"d": 64, "sigma": 0.01}


def _knob_str(arch: str, value) -> str:
    if arch == "relu_l1":
        return f"l1_{value:.0e}"  # e.g. l1_1e-04
    if arch == "jumprelu":
        return f"l0_{value:.0e}"
    if arch == "topk":
        return f"k{value:02d}"
    raise ValueError(f"unknown arch {arch}")


def _build_cell(topology: str, arch: str, value, seed: int) -> dict:
    """Return the full set of fields for one manifest row + YAML config."""
    knob_str = _knob_str(arch, value)
    run_id = f"{topology}_{arch}_{knob_str}_seed{seed}"
    config_path = CONFIGS_DIR / f"{run_id}.yaml"
    out_dir = RESULTS_DIR / topology / arch / f"{knob_str}_seed{seed}"

    sae = {**FIXED_SAE, "arch": arch, "seed": seed}
    if arch == "relu_l1":
        sae["l1_coeff"] = value
    elif arch == "topk":
        sae["k"] = int(value)
    elif arch == "jumprelu":
        sae["l0_coeff"] = value
    yaml_doc = {
        "topology": topology,
        "dgp": dict(FIXED_DGP),
        "sae": sae,
    }
    return {
        "run_id": run_id,
        "config_path": config_path,
        "out_dir": out_dir,
        "topology": topology,
        "arch": arch,
        "sparsity_knob": {"relu_l1": "l1_coeff", "topk": "k",
                          "jumprelu": "l0_coeff"}[arch],
        "sparsity_value": value,
        "seed": seed,
        "yaml_doc": yaml_doc,
    }


def _enumerate_cells():
    cells = []
    for topology in TOPOLOGIES:
        for value in L1_GRID:
            for seed in SEEDS:
                cells.append(_build_cell(topology, "relu_l1", value, seed))
        for value in K_GRID:
            for seed in SEEDS:
                cells.append(_build_cell(topology, "topk", value, seed))
        for value in L0_GRID:
            for seed in SEEDS:
                cells.append(_build_cell(topology, "jumprelu", value, seed))
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

    n_relu = sum(1 for c in cells if c["arch"] == "relu_l1")
    n_topk = sum(1 for c in cells if c["arch"] == "topk")
    n_jr = sum(1 for c in cells if c["arch"] == "jumprelu")
    print(f"wrote {len(cells)} configs to {CONFIGS_DIR}")
    print(f"  relu_l1: {n_relu}   topk: {n_topk}   jumprelu: {n_jr}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
