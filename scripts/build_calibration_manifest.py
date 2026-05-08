"""Build calibration manifest: train one SAE per (topology, arch, coeff) and
write final-L0 records to results/calibration/.

4 topologies x (10 L1 values + 9 L0 values) = 76 cells.
Wider range than the prior sweep grid so we can interpolate to hit
target L0 in {8, 16, 24, 32} per manifold.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "calibration"
LOGS_DIR = REPO / "logs" / "calibration"

TOPOLOGIES = ["circle", "helix", "sphere", "torus"]
L1_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
L0_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SEED = 0


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO / "configs" / "calibration_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for topo in TOPOLOGIES:
        for c in L1_GRID:
            rows.append((topo, "relu_l1", c))
        for c in L0_GRID:
            rows.append((topo, "jumprelu", c))

    with open(manifest_path, "w") as f:
        f.write("idx\ttopology\tarch\tcoeff\tout_path\n")
        for idx, (topo, arch, coeff) in enumerate(rows):
            run_id = f"{topo}_{arch}_c{coeff:.0e}_s{SEED}"
            out_path = RESULTS_DIR / f"{run_id}.json"
            f.write(f"{idx}\t{topo}\t{arch}\t{coeff:g}\t{out_path}\n")

    print(f"wrote {len(rows)} calibration cells to {manifest_path}")
    print(f"  relu_l1 cells: {sum(1 for r in rows if r[1] == 'relu_l1')}")
    print(f"  jumprelu cells: {sum(1 for r in rows if r[1] == 'jumprelu')}")


if __name__ == "__main__":
    main()
