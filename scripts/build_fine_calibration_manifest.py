"""Fine calibration pass — manifold-specific grids that densely sample the
sparsity range producing L0 in roughly [5, 40].

Coeff choices come from the first-pass calibration (results/calibration/summary.csv):
each grid focuses on the narrow coefficient window where L0 sweeps through
the target [8, 32] band on that (manifold, arch) pair.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "calibration"   # same dir, new filenames
LOGS_DIR = REPO / "logs" / "calibration_fine"

# Manifold-specific fine grids
GRIDS = {
    ("circle", "relu_l1"):  [3e-4, 7e-4, 1.5e-3, 3e-3, 5e-3, 1e-2, 1.5e-2, 2e-2],
    ("circle", "jumprelu"): [5e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 7e-4],

    ("helix", "relu_l1"):   [3e-2, 5e-2, 7e-2, 1e-1, 1.5e-1, 2e-1],
    ("helix", "jumprelu"):  [3e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1.0, 2.0],

    ("sphere", "relu_l1"):  [3e-4, 7e-4, 1.5e-3, 3e-3, 5e-3, 1e-2, 1.5e-2, 2e-2],
    ("sphere", "jumprelu"): [3e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4],

    ("torus", "relu_l1"):   [7e-4, 1.5e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2],
    ("torus", "jumprelu"):  [3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3],
}
SEED = 0


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO / "configs" / "calibration_fine_manifest.tsv"
    rows = []
    for (topo, arch), coeffs in GRIDS.items():
        for c in coeffs:
            rows.append((topo, arch, c))

    with open(manifest_path, "w") as f:
        f.write("idx\ttopology\tarch\tcoeff\tout_path\n")
        for idx, (topo, arch, coeff) in enumerate(rows):
            run_id = f"{topo}_{arch}_c{coeff:.0e}_s{SEED}"
            out_path = RESULTS_DIR / f"{run_id}.json"
            f.write(f"{idx}\t{topo}\t{arch}\t{coeff:g}\t{out_path}\n")

    print(f"wrote {len(rows)} fine-calibration cells to {manifest_path}")
    n_rl = sum(1 for r in rows if r[1] == "relu_l1")
    n_jr = sum(1 for r in rows if r[1] == "jumprelu")
    print(f"  relu_l1: {n_rl}  jumprelu: {n_jr}")


if __name__ == "__main__":
    main()
