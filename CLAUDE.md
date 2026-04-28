# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Topological Inductive Biases of Sparse Autoencoders** — does the topology of a known source manifold $M$ survive an SAE's encoding, and does survival depend systematically on the SAE architecture (ReLU+L1 / TopK / JumpReLU)? See `sae_topology_project.md` for the full spec.

The project works entirely with **synthetic toy manifolds** ($S^1, T^2, S^2$, figure-8) where ground-truth topology and the closed-form Laplace–Beltrami spectrum are known. Real-LM features are out of scope.

## Conceptual Framework

Given an SAE with encoder $W_e$, decoder $W_d$, nonlinearity $\sigma$:
- **Pre-activations:** $z_\text{pre}(x) = W_e x + b_e$
- **Post-activations** (primary object): $z(x) = \sigma(z_\text{pre}(x))$
- **Reconstructions:** $\hat x(x) = W_d z(x) + b_d$

For each manifold sample, we measure topology preservation under post-activations using two complementary tools (spec §4):

1. **Mapper** with three filter functions (Laplacian eigenvector / ground-truth / PCA) → "is the topological type preserved?" via Mapper-graph Betti numbers $(b_0, b_1)$.
2. **Coifman–Lafon $\alpha=1$ graph Laplacian** → "is the metric shape preserved?" via log-ratio error $\mathcal{E}$ against the closed-form Laplace–Beltrami spectrum.

**Headline hypothesis (H1):** for a TopK SAE on a $d$-manifold, recovery requires $K \geq d+1$ — sharp phase transition at $K = d+1$ as the SAE's per-sample feature budget crosses the manifold's intrinsic dimension.

## Repository Structure

```
sae_topology/
  dgp/                # data-generating processes (Circle, Torus, Sphere, FigureEight, ...)
    base.py           # TopologicalSpace ABC with sample_with_gt(n)
    spaces.py         # all manifold classes; gt_coords surfaced for Circle, Torus, Sphere
  saes/               # SAE architectures and trainer
    architectures.py  # ReluL1SAE, TopKSAE, JumpReLUSAE — forward returns 'pre' key
    trainer.py        # online sampling, dead-atom resampling, unit-norm decoder
    config.py         # SAETrainConfig (incl. K_sweep field)
  mapper/             # Mapper pipeline
    filters.py        # laplacian_eigenvector_filter, ground_truth_filter, pca_filter
    pipeline.py       # mapper_sweep, FirstGapAgglomerative clusterer, graph_betti (with 2-cells), correct_region
  spectral/           # Coifman–Lafon graph Laplacian + reference LB spectra
    graph_laplacian.py
    metrics.py        # log_ratio_error, multiplicity_check, near_zero_count
    reference.py      # closed-form S^1, T^2, S^2 spectra; GROUND_TRUTH_BETTI
  experiments/
    runner.py         # ExperimentResult dataclass + run_sae_experiment; reads Stage 0 baseline
    sweep.py          # width_sweep, architecture_sweep, topk_k_sweep
    stage0_validate.py # raw-sample validation + auto-tune (knn_k, sigma_factor); writes baseline.yaml
  analysis/
    plot.py           # plot_dgp_samples, plot_training_curves, plot_mapper_graph, plot_spectrum_vs_theory, plot_topk_phase_transition
    aggregate.py      # load_results -> DataFrame; summary table; TopK phase figure; per-run Mapper + spectrum figures
  configs/
    stage0/baseline.yaml      # per-manifold tuned (knn_k, sigma_factor) — ground truth that Stage 1+ reads
    stage1_smoke_circle.yaml  # ReLU+L1 on S^1 smoke test
    topk_sweep.yaml           # H1 phase-transition driver
  results/            # per-run output dirs: pre.npy, post.npy, recon.npy, mapper.json, spectral.json, result.json
```

## Stages

| Stage | Purpose | Entry point | Output |
|---|---|---|---|
| 0 | Validate Mapper + spectral pipeline on raw samples; auto-tune `(knn_k, sigma_factor)` per manifold | `python -m sae_topology.experiments.stage0_validate` | `configs/stage0/baseline.yaml` |
| 1 | Train an SAE on one (manifold, arch) configuration | `python -m sae_topology.experiments.runner --config <yaml>` | `results/.../{pre,post,recon}.npy + result.json + mapper.json + spectral.json` |
| 2 | Topology + spectral analysis on post-activations (continuation of Stage 1; uses Stage-0 tuned `knn_k, sigma_factor`) | (inside runner) | same files |
| 3 | Aggregate across runs; figures + summary CSV | `python -m sae_topology.analysis.aggregate --results_dir <dir> --out figures/` | `summary.csv`, `summary_table.png`, `topk_phase_<m>.png`, optional per-run Mapper + spectrum figures with `--per_run` |
| H1 sweep | TopK k phase-transition runs | `python -m sae_topology.experiments.sweep --topk-k --manifold <m> --K 1,2,3,4,6,8` | runs saved into `results/topk_sweep/` |

**Stage 0 must pass before Stage 1+ results are interpreted.** `baseline.yaml` per-manifold pass criterion: correct contiguous region of the (n_intervals, overlap) grid that matches GT Betti is ≥ 50% of the 20-config grid; spectral $\mathcal{E} < 0.05$ on raw samples.

## Key Design Decisions

- **Post-activations are the primary object.** The dictionary alone is topologically uninformative (e.g. a 4-atom "cross" representation of $S^1$ shows nothing of the loop). Spec §2.2.
- **Mapper-as-simplicial-complex Betti** (not graph Euler): three nodes with a shared point form a filled triangle; we compute $b_1$ via simplicial homology over GF(2), not $E - V + b_0$. Without this fix, high-overlap covers gave $b_1$ inflated 5–30× in tests.
- **Single-linkage with global threshold.** "First gap on the dendrogram" (the spec's literal proposal) over-fragmented dense manifold data because of the chain-effect long tail of merge heights. We instead use $5 \times \text{median}(k\text{NN distance})$ from the full dataset as a fixed cluster-merge threshold; it produces 1 cluster per cover box for connected manifolds, and only fragments when there's a true bridge.
- **Coifman–Lafon $\alpha = 1$ normalization** removes the sampling-density correction term, giving the pure $\Delta_g$ spectrum independent of finite-$N$ density variation.
- **Existing DGP keeps random-orthonormal-frame embedding** rather than spec-literal zero-padding. The two are equivalent up to a fixed orthogonal rotation, which is invisible to Mapper, the graph Laplacian, and the optimal SAE; finite-step SGD may differ marginally but multi-seed runs absorb it. Explicitly chosen for marginally better residual-stream realism.
- **Stage 0 auto-tunes** $(k_\text{nn}, \sigma_\text{factor})$ per manifold over $\{15, 25, 40\} \times \{0.5, 1.0, 2.0\}$, picking the config that maximizes the correct-region fraction (tie-break by spectral $\mathcal{E}$). Stage 1+ reads `baseline.yaml` and uses these per-manifold tuned values.
- **`correct_region` not `stable_region`**: the spec's "headline = stable region" was modal-Betti and not correctness-aware. In Stage 0/1 we know GT and use the largest 4-connected region of the grid where Betti matches GT.

## Key Dependencies

`numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `networkx`, `kmapper`, `pandas`, `pyyaml`.

## Known Risks

- **Random-init SAEs may not fail on simple manifolds.** With ReLU and $m \gg d$, random projections of the manifold preserve topology by Johnson–Lindenstrauss-style arguments. Smoke tests on random-init ReLU+L1 on $S^1$ recover $(1, 1)$ Betti and $\mathcal{E} \approx 0.02$. The spec's "random-init negative control" is therefore a weaker check than implied; undersized-dictionary or pre-trained-but-collapsed controls are stricter.
- **TopK collapses post-activations into many components.** Empirically, TopK at $k \in \{1, 2, 3\}$ on $S^1$ at 10k training steps gives $b_0 \in [30, 90]$ while MSE is low — the SAE encodes the manifold via a discrete partition of feature triplets rather than a smooth ring. Whether this resolves at 30k+ steps is open. Treat short-run TopK results with skepticism.
- **Cover-overlap pollution at high overlap** (`overlap=0.5`): triple-overlap cover regions inflate $b_1$ unless 2-cells are filled. Our `graph_betti` does this; without the fix, $b_1$ would be reported 5–30× too high.
- **Stage 0 tuning runtime:** auto-sweep over 9 configs × 4 manifolds with $N \in \{2000, 10000\}$ takes ~30–60 minutes serial. Use `--n_jobs N` (joblib loky) to parallelize Stage A (12 jobs: 4 topo × 3 knn_k, with 3 sigma_factor variants sharing kNN inside each job) and Stage B (≤12 jobs: topo × filter); 12 workers is the natural cap. Use `--quick` flag for development at half-N. Determinism: serial vs parallel produces bit-identical `baseline.yaml` (covered in `tests/test_stage0_determinism.py`).

## Parallelism

Single `--n_jobs N` flag toggles process-level parallelism (joblib loky backend, BLAS pinned to 1 thread per worker via `threadpoolctl`):

- `mapper_sweep(..., n_jobs=N)`: dispatches the (n_intervals × overlap) cover configs across N workers. Importers calling from inside an outer joblib pool MUST pass `n_jobs=1` (no nested loky pools).
- `python -m sae_topology.experiments.stage0_validate --n_jobs N`: flat (topology × knn_k) Stage A pool + (topology × filter) Stage B pool, both at level N.
- `python -m sae_topology.experiments.runner --config <yaml> --n_jobs N`: Mapper (Lap / GT / PCA) filters dispatched in parallel.
- `python -m sae_topology.experiments.measure_parallel_speedup --jobs 1 4 12`: benchmark wall-clock at varying N, with `--check_identical` to confirm correctness.

The full test matrix lives in `tests/` (24 tests, ~8 min on the 56-core compute node):
- `test_knn_cache.py` — `precomputed_knn` API for sigma_factor variant sharing.
- `test_sparse_eigsh_robustness.py` — 4-tier ARPACK fallback cascade in `laplacian_eigendecomposition`.
- `test_parallel_determinism.py` — `mapper_sweep` serial == parallel.
- `test_runner_filter_parallelism.py` — runner.py serial == parallel.
- `test_stage0_determinism.py` — full Stage 0 pipeline serial == parallel (incl. byte-equal `baseline.yaml`).

## Stage 0 Pass / Fail Conventions

Per manifold, the runner expects `configs/stage0/baseline.yaml` to specify the tuned `(knn_k, sigma_factor)` and to have `overall_pass: true`. If the baseline file is missing or a manifold isn't in it, the runner falls back to defaults (`knn_k=25, sigma_factor=1.0`) and prints a warning.

The pass threshold is 50% of the 20-config Mapper grid in a contiguous correct-Betti region (`correct_region_fraction >= 0.5`), plus spectral $\mathcal{E} < 0.05$. These are tunable in `stage0_validate.stage0_for_topology`.
