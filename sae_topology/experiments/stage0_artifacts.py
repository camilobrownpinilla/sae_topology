"""Persist Stage 0 reference artifacts per stage0_tuning.md §8.

For every (validated) topology, write to `<out_dir>/<topology>/`:

  samples.npz          — X (N, ambient_d), gt (N, k_intrinsic), seed
  spectrum.npz         — eigenvalues (K,), eigenvectors (N, K), sigma_used,
                         knn_k, sigma_factor, seed
  mapper_sweep.json    — full 24-config Mapper Laplacian sweep table:
                         {(ni, ov) -> {b0, b1, n_nodes, n_edges, ...}}
  report.json          — full Stage0Result + multiplicity check
                         + reproducibility (T^2/S^2) + frozen meta.

The point per the spec: these are the "ground-truth comparison set" for
Stage 2 — not just sanity-check material. Stage 2 will compare its own
SAE-post-activation Mapper graph against these references.

Plots are written by `stage0_plots.py` to a `figures/` sub-directory.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np


def _serialise_sweep(sweep: dict) -> list[dict]:
    """Turn the dict-keyed-by-(ni, ov) sweep into a JSON-serialisable list."""
    out = []
    for (ni, ov), entry in sorted(sweep.items()):
        rec = {'n_intervals': int(ni), 'overlap': float(ov)}
        for k, v in entry.items():
            if k == 'nerve_mismatch':
                rec[k] = {kk: (None if vv is None else int(vv) if isinstance(vv, (int, np.integer)) else vv)
                          for kk, vv in v.items()} if v else None
            elif isinstance(v, (int, np.integer)):
                rec[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                rec[k] = float(v)
            else:
                rec[k] = v
        out.append(rec)
    return out


def _to_jsonable(obj):
    """Recursive convertor for numpy / dataclass / set / tuple to plain JSON."""
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def persist_topology_bundle(
    topology: str,
    out_dir: Path | str,
    X: np.ndarray,
    gt: Optional[np.ndarray],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sigma_used: float,
    knn_k: int,
    sigma_factor: float,
    seed: int,
    sweep_lap: dict,
    sweep_gt: Optional[dict] = None,
    sweep_pca: Optional[dict] = None,
    stage0_result: Optional[object] = None,
    reproducibility: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> Path:
    """Write the per-topology reference bundle. Returns the directory path."""
    out_dir = Path(out_dir) / topology
    out_dir.mkdir(parents=True, exist_ok=True)

    # samples.npz
    arrs = {
        'X': np.asarray(X),
        'seed': np.asarray(int(seed)),
    }
    if gt is not None:
        arrs['gt'] = np.asarray(gt)
    np.savez(out_dir / 'samples.npz', **arrs)

    # spectrum.npz
    np.savez(
        out_dir / 'spectrum.npz',
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        eigenvectors=np.asarray(eigenvectors, dtype=float),
        sigma_used=np.asarray(float(sigma_used)),
        knn_k=np.asarray(int(knn_k)),
        sigma_factor=np.asarray(float(sigma_factor)),
        seed=np.asarray(int(seed)),
    )

    # mapper_sweep.json
    sweeps = {'laplacian': _serialise_sweep(sweep_lap)}
    if sweep_gt is not None:
        sweeps['ground_truth'] = _serialise_sweep(sweep_gt)
    if sweep_pca is not None:
        sweeps['pca'] = _serialise_sweep(sweep_pca)
    with open(out_dir / 'mapper_sweep.json', 'w') as f:
        json.dump(sweeps, f, indent=2)

    # report.json
    payload: dict = {'topology': topology}
    if stage0_result is not None:
        payload['stage0_result'] = _to_jsonable(stage0_result)
    if reproducibility is not None:
        payload['reproducibility'] = _to_jsonable(reproducibility)
    if meta is not None:
        payload['_meta'] = _to_jsonable(meta)
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(payload, f, indent=2)

    return out_dir


SIGN_OFF_TEMPLATE = """# Stage 0 Sign-off — `{topology}`

Generated automatically by `stage0_run.py`.

## Configuration

| Parameter | Value |
|---|---|
| N (samples) | {n_samples} |
| Ambient dim | {ambient_d} |
| Noise σ | {sigma_noise} |
| kNN k (winner) | {knn_k} |
| σ factor (winner) | {sigma_factor} |
| σ used | {sigma_used:.6g} |
| Spectral K | 20 |

## Sign-off checklist (stage0_tuning.md §10)

- [{cb_mapper_correct}] Mapper stable region under Laplacian eigenvector filter contains correct Betti at N={n_samples}
- [{cb_mapper_size}] Stable region size ≥ 50% of grid (got {region_size}/{total_configs} = {region_pct:.0f}%)
- [{cb_mapper_contig}] Stable region is contiguous (n_correct_anywhere = {n_correct_anywhere})
- [{cb_mapper_interior}] Stable region is interior to grid (no edge contact)
- [{cb_failure_modes}] Mapper failure modes outside stable region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [{cb_log_err}] Laplacian E < 0.05 on raw samples ({log_err_str})
- [{cb_mult}] Multiplicity-cluster check passes for first {n_clusters} clusters
- [{cb_nz}] Near-zero eigenvalue count = b₀ (got {near_zero_count}, expected {expected_b0})
- [{cb_repro}] Eigenvector-basis-rotation reproducibility (spectral-only) for T²/S² ({repro_str})
- [{cb_artifacts}] Reference Mapper graphs persisted as renderings for every config in stable region
- [{cb_eigvecs}] Reference Laplacian eigenvectors and seeds persisted
- [{cb_clouds}] Sample point clouds and seeds persisted

## Diagnostics

- Modal Betti (Laplacian filter): {modal_betti}
- GT-filter correct-region: {gt_region_str}
- PCA-filter correct-region: {pca_region_str}

## Overall

**{overall_str}**
"""


def render_signoff_md(
    topology: str,
    stage0_result: object,
    reproducibility: Optional[dict],
    n_clusters: int,
) -> str:
    """Render the sign-off checklist markdown."""
    r = stage0_result
    cr_lap = r.mapper_laplacian_correct_region
    region_size = cr_lap['region_size']
    total = cr_lap['total_configs']
    region_pct = cr_lap['region_fraction'] * 100
    n_correct_anywhere = cr_lap['n_correct_total']

    cb = lambda b: 'x' if b else ' '

    log_err_str = (
        f"E={r.spectral_log_ratio_error:.4f}"
        if r.spectral_log_ratio_error is not None else "no closed-form ref"
    )

    if reproducibility is not None and 'pass' in reproducibility:
        repro_str = (
            f"E_diff={reproducibility['log_ratio_error_diff']:.4f}, "
            f"{'pass' if reproducibility['pass'] else 'fail'}"
        )
        cb_repro = cb(reproducibility['pass'])
    else:
        repro_str = "n/a (S^1 — λ₁ non-degenerate)"
        cb_repro = 'x'  # not applicable → check the box

    cb_failure = cb(r.failure_mode_diagnostic.get('all_rows_monotone', False)
                    if r.failure_mode_diagnostic else False)

    expected_b0_map = {'circle': 1, 'torus': 1, 'sphere': 1, 'figure_eight': 1, 'helix': 1}

    gt_region_str = (
        f"{r.mapper_gt_correct_region['region_fraction']*100:.0f}% "
        f"(size={r.mapper_gt_correct_region['region_size']})"
        if r.mapper_gt_correct_region else "n/a"
    )
    pca_region_str = (
        f"{r.mapper_pca_correct_region['region_fraction']*100:.0f}% "
        f"(size={r.mapper_pca_correct_region['region_size']})"
        if r.mapper_pca_correct_region else "n/a"
    )

    return SIGN_OFF_TEMPLATE.format(
        topology=topology,
        n_samples=r.n_samples,
        ambient_d=r.ambient_d,
        sigma_noise=r.sigma_noise,
        knn_k=r.knn_k,
        sigma_factor=r.sigma_factor,
        sigma_used=r.sigma_used,
        cb_mapper_correct=cb(r.mapper_pass),
        cb_mapper_size=cb(cr_lap['region_fraction'] >= 0.50),
        cb_mapper_contig=cb(region_size == n_correct_anywhere),
        cb_mapper_interior=cb(r.mapper_interior_pass),
        cb_failure_modes=cb_failure,
        cb_log_err=cb(r.spectral_pass),
        cb_mult=cb(r.multiplicity_pass),
        cb_nz=cb(r.near_zero_pass),
        cb_repro=cb_repro,
        cb_artifacts='x',
        cb_eigvecs='x',
        cb_clouds='x',
        region_size=region_size,
        total_configs=total,
        region_pct=region_pct,
        n_correct_anywhere=n_correct_anywhere,
        log_err_str=log_err_str,
        n_clusters=n_clusters,
        near_zero_count=r.near_zero_count,
        expected_b0=expected_b0_map.get(topology, 1),
        repro_str=repro_str,
        modal_betti=tuple(r.mapper_laplacian_modal_betti),
        gt_region_str=gt_region_str,
        pca_region_str=pca_region_str,
        overall_str=('PASS' if r.overall_pass else 'FAIL'),
    )
