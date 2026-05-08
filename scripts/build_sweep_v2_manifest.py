"""Build sweep_v2 manifest using calibrated coefficients.

For each (topology, arch), four sparsity coefficients selected to land
the per-sample L0 near {8, 16, 24, 32}. Each (topo, arch, target) gets 3 seeds.
Total: 4 topo x 2 arch x 4 targets x 3 seeds = 96 cells.

Picks come from results/calibration/ (50k-step single-seed calibration sweep,
distinct-coeff-per-target greedy assignment).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO / "configs" / "sweep_v2"
RESULTS_DIR = REPO / "results" / "sweep_v2"
LOGS_DIR = REPO / "logs" / "sweep_v2"
CALIB_DIR = REPO / "results" / "calibration"

TOPOLOGIES = ["circle", "helix", "sphere", "torus"]
ARCHS = ["relu_l1", "jumprelu"]
TARGETS = [8, 16, 24, 32]
SEEDS = [0, 1, 2]

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


def _select_picks() -> dict:
    """Re-derive (topology, arch, target) -> (coeff, achieved_L0) from calibration data,
    using a distinct-coeff-per-target greedy assignment."""
    import pandas as pd
    rows = []
    for p in sorted(CALIB_DIR.glob("*.json")):
        if p.name in {"picks.json", "final_picks.json"}: continue
        with open(p) as f: d = json.load(f)
        rows.append({
            "topo": d["topology"], "arch": d["arch"], "coeff": d["coeff"],
            "L0": d["mean_l0_tail"], "n_dead": d["n_dead_final"],
            "mse": d["mse_tail"],
        })
    df = pd.DataFrame(rows)
    picks = {}
    for topo in TOPOLOGIES:
        for arch in ARCHS:
            s = df[(df.topo == topo) & (df.arch == arch)
                   & (df.L0 > 1) & (df.n_dead < 128)].copy()
            used = set()
            for tgt in TARGETS:
                avail = s[~s.coeff.isin(used)].copy()
                if avail.empty:
                    raise RuntimeError(f"no candidate for {topo}/{arch}/{tgt}")
                avail["gap"] = (avail.L0 - tgt).abs()
                best = avail.loc[avail.gap.idxmin()]
                used.add(float(best.coeff))
                picks[(topo, arch, tgt)] = (
                    float(best.coeff), float(best.L0), float(best.mse),
                )
    return picks


def _knob_str(target: int) -> str:
    return f"t{target:02d}"


def _coeff_field(arch: str) -> str:
    return {"relu_l1": "l1_coeff", "jumprelu": "l0_coeff"}[arch]


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    picks = _select_picks()

    rows = []
    for topo in TOPOLOGIES:
        for arch in ARCHS:
            for tgt in TARGETS:
                coeff, achieved_L0, mse = picks[(topo, arch, tgt)]
                for seed in SEEDS:
                    knob_str = _knob_str(tgt)
                    run_id = f"{topo}_{arch}_{knob_str}_seed{seed}"
                    config_path = CONFIGS_DIR / f"{run_id}.yaml"
                    out_dir = RESULTS_DIR / topo / arch / f"{knob_str}_seed{seed}"

                    sae = {**FIXED_SAE, "arch": arch, "seed": seed,
                           _coeff_field(arch): coeff}
                    yaml_doc = {
                        "topology": topo,
                        "dgp": dict(DGP_PARAMS[topo]),
                        "sae": sae,
                    }
                    with open(config_path, "w") as f:
                        yaml.safe_dump(yaml_doc, f, sort_keys=False)
                    rows.append((run_id, config_path, out_dir, topo, arch,
                                 tgt, coeff, achieved_L0, seed))

    manifest_path = CONFIGS_DIR / "manifest.tsv"
    header = ["idx", "run_id", "config_path", "out_dir", "topology", "arch",
              "target_L0", "coeff", "calib_achieved_L0", "seed"]
    with open(manifest_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for idx, (rid, cp, od, t, a, tgt, c, l0, s) in enumerate(rows):
            f.write("\t".join([
                str(idx), rid, str(cp), str(od), t, a,
                str(tgt), f"{c:g}", f"{l0:.2f}", str(s),
            ]) + "\n")

    print(f"wrote {len(rows)} configs to {CONFIGS_DIR}")
    print(f"manifest: {manifest_path}")
    print(f"picks summary:")
    print(f"  {'topo':<8s} {'arch':<10s} {'target':>6s} {'coeff':>10s} {'calib_L0':>10s}")
    for (topo, arch, tgt), (coeff, l0, mse) in sorted(picks.items()):
        print(f"  {topo:<8s} {arch:<10s} {tgt:>6d} {coeff:>10.2e} {l0:>10.2f}")


if __name__ == "__main__":
    main()
