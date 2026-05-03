"""Aggregate sweep_v1 results: walk report.json files -> summary.csv + figures.

Reads `configs/sweep_v1/manifest.tsv` to enumerate expected runs, joins each
row with `<out_dir>/report.json` if present, and produces:

    summary.csv                — per-run flat row
    recovery_vs_sparsity.png   — 3x3 grid (topology x arch); correctness +
                                 log-ratio error vs sparsity knob
    topk_phase_transition.png  — 3-panel correctness vs k, with vertical line
                                 at k = d_intrinsic + 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs" / "sweep_v1" / "manifest.tsv"
DEFAULT_RESULTS = REPO / "results" / "sweep_v1"

# Intrinsic dimension per topology, for the H1 phase-transition line.
D_INTRINSIC = {"circle": 1, "torus": 2, "sphere": 2}

ARCH_ORDER = ["relu_l1", "topk", "jumprelu"]
TOPO_ORDER = ["circle", "torus", "sphere"]


def _load_report(out_dir: Path) -> dict | None:
    p = out_dir / "report.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _flatten(report: dict) -> dict:
    """Pull the columns we care about out of a report.json."""
    r = report.get("stage1_result", {})
    cr = r.get("mapper_laplacian_correct_region", {}) or {}
    modal = r.get("mapper_laplacian_modal_betti", [None, None])
    return {
        "spectral_log_ratio_error": r.get("spectral_log_ratio_error"),
        "region_fraction": cr.get("region_fraction"),
        "region_size": cr.get("region_size"),
        "is_interior": cr.get("is_interior"),
        "modal_b0": modal[0] if len(modal) > 0 else None,
        "modal_b1": modal[1] if len(modal) > 1 else None,
        "spectral_pass": r.get("spectral_pass"),
        "mapper_pass": r.get("mapper_pass"),
        "multiplicity_pass": r.get("multiplicity_pass"),
        "near_zero_pass": r.get("near_zero_pass"),
        "overall_pass": r.get("overall_pass"),
        "final_mse": r.get("final_mse"),
        "final_mean_l0": r.get("final_mean_l0"),
        "final_n_dead": r.get("final_n_dead"),
        "knn_k": r.get("knn_k"),
        "sigma_factor": r.get("sigma_factor"),
    }


def build_summary(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    rows = []
    for _, m in manifest.iterrows():
        out_dir = Path(m["out_dir"])
        report = _load_report(out_dir)
        row = m.to_dict()
        row["found"] = report is not None
        if report is not None:
            row.update(_flatten(report))
        rows.append(row)
    return pd.DataFrame(rows)


def _agg_seeds(group: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, means, stds) across seeds, sorted by sparsity_value."""
    if col not in group.columns or "sparsity_value" not in group.columns:
        return np.array([]), np.array([]), np.array([])
    g = group.dropna(subset=[col, "sparsity_value"])
    if g.empty:
        return np.array([]), np.array([]), np.array([])
    grouped = g.groupby("sparsity_value")[col]
    xs = np.array(sorted(grouped.groups.keys()))
    means = np.array([grouped.get_group(x).mean() for x in xs])
    stds = np.array([grouped.get_group(x).std(ddof=1) for x in xs])
    stds = np.nan_to_num(stds, nan=0.0)
    return xs, means, stds


def plot_recovery_vs_sparsity(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex='col')

    for r, topo in enumerate(TOPO_ORDER):
        for c, arch in enumerate(ARCH_ORDER):
            ax = axes[r, c]
            sub = df[(df["topology"] == topo) & (df["arch"] == arch) & df["found"]]

            x_corr, m_corr, s_corr = _agg_seeds(sub, "region_fraction")
            x_err, m_err, s_err = _agg_seeds(sub, "spectral_log_ratio_error")

            log_x = (arch != "topk")
            if log_x:
                ax.set_xscale("log")

            # Left axis: correctness (region_fraction)
            if len(x_corr) > 0:
                ax.errorbar(x_corr, m_corr, yerr=s_corr, fmt="o-",
                            color="tab:blue", label="correct region", capsize=3)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="tab:blue", linestyle=":", alpha=0.4)
            ax.set_ylabel("correct region fraction", color="tab:blue")
            ax.tick_params(axis='y', labelcolor="tab:blue")

            # Right axis: spectral log-ratio error
            ax2 = ax.twinx()
            if len(x_err) > 0:
                # Cap visualisation at 5x threshold to avoid blow-up dominating axis.
                m_err_clipped = np.clip(m_err, 0, 0.5)
                ax2.errorbar(x_err, m_err_clipped, yerr=s_err, fmt="s--",
                             color="tab:red", label="log-ratio error", capsize=3)
            ax2.set_ylim(0, 0.5)
            ax2.axhline(0.05, color="tab:red", linestyle=":", alpha=0.4)
            ax2.set_ylabel("spectral E", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")

            # H1 phase-transition vertical line for topk
            if arch == "topk":
                k_star = D_INTRINSIC[topo] + 1
                ax.axvline(k_star, color="black", linestyle="--", alpha=0.5,
                           label=f"k=d+1={k_star}")
                ax.legend(loc="lower right", fontsize=8)

            knob = {"relu_l1": "l1_coeff", "topk": "k",
                    "jumprelu": "l0_coeff"}[arch]
            ax.set_xlabel(knob)
            if r == 0:
                ax.set_title(f"{arch}", fontsize=11)
            if c == 0:
                ax.text(-0.27, 0.5, topo, transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=12,
                        fontweight='bold')

    fig.suptitle("Sweep v1 — Recovery vs sparsity knob (mean ± std across 3 seeds)",
                 fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_topk_phase_transition(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    sub = df[(df["arch"] == "topk") & df["found"]]

    for ax, topo in zip(axes, TOPO_ORDER):
        s = sub[sub["topology"] == topo]
        x, m, e = _agg_seeds(s, "region_fraction")
        if len(x) > 0:
            ax.errorbar(x, m, yerr=e, fmt="o-", color="tab:blue",
                        label="correct region", capsize=3)

        # spectral on twin axis
        ax2 = ax.twinx()
        x2, m2, e2 = _agg_seeds(s, "spectral_log_ratio_error")
        if len(x2) > 0:
            m2_clipped = np.clip(m2, 0, 0.5)
            ax2.errorbar(x2, m2_clipped, yerr=e2, fmt="s--", color="tab:red",
                         label="log-ratio error", capsize=3)
        ax2.set_ylim(0, 0.5)
        ax2.axhline(0.05, color="tab:red", linestyle=":", alpha=0.4)
        if topo == TOPO_ORDER[-1]:
            ax2.set_ylabel("spectral E", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")

        k_star = D_INTRINSIC[topo] + 1
        ax.axvline(k_star, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{topo} (d={D_INTRINSIC[topo]}, k*={k_star})")
        ax.set_xlabel("k")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="tab:blue", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("correct region fraction", color="tab:blue")
    axes[0].tick_params(axis='y', labelcolor="tab:blue")
    fig.suptitle("TopK phase transition: recovery vs k", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--out", default=str(DEFAULT_RESULTS))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = build_summary(Path(args.manifest))
    csv_path = out / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({df['found'].sum()}/{len(df)} runs found)")

    plot_recovery_vs_sparsity(df, out / "recovery_vs_sparsity.png")
    print(f"wrote {out / 'recovery_vs_sparsity.png'}")
    plot_topk_phase_transition(df, out / "topk_phase_transition.png")
    print(f"wrote {out / 'topk_phase_transition.png'}")


if __name__ == "__main__":
    main()
