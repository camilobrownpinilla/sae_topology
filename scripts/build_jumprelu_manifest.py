"""Build the sweep_jumprelu manifest: 60 per-run YAML configs + manifest.tsv.

Sweep matrix:
    4 topologies (circle, helix, sphere, torus)
    x 5 l0_coeff values: {1e-3, 2e-3, 3e-3, 5e-3, 7e-3}
    x 3 seeds (0, 1, 2)
    = 60 runs

Targets the L0 ∈ [8, 32] window (knee between l0_coeff = 1e-3 and 1e-2).

Output dirs go into the unified results/sweep_v1/<topo>/jumprelu/ namespace
so the existing aggregation tooling discovers them alongside circle/sphere/torus
relu_l1/topk and helix relu_l1/topk runs.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO / "configs" / "sweep_jumprelu"
RESULTS_DIR = REPO / "results" / "sweep_v1"
LOGS_DIR = REPO / "logs" / "sweep_jumprelu"

TOPOLOGIES = ["circle", "helix", "sphere", "torus"]
SEEDS = [0, 1, 2]
L0_GRID = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3]

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

DGP_PARAMS = {
    "circle": {"d": 64, "sigma": 0.01},
    "sphere": {"d": 64, "sigma": 0.01},
    "torus":  {"d": 64, "sigma": 0.01},
    "helix":  {"d": 64, "sigma": 0.01, "radius": 1.0, "pitch": 0.5, "n_turns": 4.0},
}


def _knob_str(value: float) -> str:
    return f"l0_{value:.0e}"


def _build_cell(topology: str, value: float, seed: int) -> dict:
    knob_str = _knob_str(value)
    run_id = f"{topology}_jumprelu_{knob_str}_seed{seed}"
    config_path = CONFIGS_DIR / f"{run_id}.yaml"
    out_dir = RESULTS_DIR / topology / "jumprelu" / f"{knob_str}_seed{seed}"

    sae = {**FIXED_SAE, "arch": "jumprelu", "seed": seed, "l0_coeff": value}
    yaml_doc = {
        "topology": topology,
        "dgp": dict(DGP_PARAMS[topology]),
        "sae": sae,
    }
    return {
        "run_id": run_id,
        "config_path": config_path,
        "out_dir": out_dir,
        "topology": topology,
        "arch": "jumprelu",
        "sparsity_knob": "l0_coeff",
        "sparsity_value": value,
        "seed": seed,
        "yaml_doc": yaml_doc,
    }


def _enumerate_cells():
    cells = []
    for topology in TOPOLOGIES:
        for value in L0_GRID:
            for seed in SEEDS:
                cells.append(_build_cell(topology, value, seed))
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

    print(f"wrote {len(cells)} configs to {CONFIGS_DIR}")
    print(f"manifest: {manifest_path}")
    by_topo = {}
    for c in cells:
        by_topo.setdefault(c["topology"], 0)
        by_topo[c["topology"]] += 1
    for t, n in by_topo.items():
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
