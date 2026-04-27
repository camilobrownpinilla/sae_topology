# SAE Topological Recoverability

## Goal

Characterize which topologies different SAE architectures can recover from activation data, and in what representational form. The project tests the hypothesis that architectural inductive bias in SAEs determines not just *which* features are recovered but *how* they are encoded — as atom-located structure (T1) or as code-located structure (T2).

## Core conceptual framework

Given a trained SAE with decoder $W_{\text{dec}} \in \mathbb{R}^{d \times m}$ (columns $d_1, \ldots, d_m$, the "atoms") producing sparse codes $z(x) \in \mathbb{R}^m$ for input $x$, two distinct objects can carry the topology of the underlying feature manifold $M$:

- **T1 (atom-located):** Each atom $d_i$ is itself a feature; the topology of $M$ lives in the geometric arrangement of atoms in $\mathbb{R}^d$. Measured by persistent homology (PH) of the decoder point cloud $\{d_i\}$.
- **T2 (code-located):** Atoms are coordinate axes spanning a subspace; features are points in code space, and the topology of $M$ lives in the point cloud of codes $\{z(x) : x \in \text{data}\}$. Measured by PH of the (typically PCA-reduced) code point cloud.

The Engels et al. (2024) days-of-the-week circle is a known T2 phenomenon in GPT-2. Standard SAE analyses look at decoder atoms (T1) and may miss T2-form recoveries entirely.

## Hypothesis

Different SAE architectures systematically prefer T1 vs. T2 representations of the same underlying $M$. Mechanistically:

- **ReLU + L1**: L1 penalizes total activation mass; T1 (1 active atom per sample) costs less than T2 (multiple coactive atoms). Should prefer T1.
- **TopK** ($k > 1$): Allows coactivation, opens T2 as a viable solution. Behavior depends on width and training dynamics.
- **JumpReLU**: Thresholding without smooth coactivation penalty; should be more T2-permissive than ReLU + L1.

## Experimental design

### Factorial structure

| Axis | Values |
|------|--------|
| Topology | $k$ isolated points; $S^1$; two disjoint $S^1$; figure-8 (wedge of two circles) |
| Architecture | ReLU + L1, TopK (class report); JumpReLU (publication extension) |
| SAE width $m$ | Span under-, matched-, overcomplete relative to target live-atom count |
| Seeds | $\geq 3$ per cell |

### Data generating process

For each topology $T$, sample $N \sim 10^4$–$10^5$ points densely on a manifold $M$ of topology $T$, embedded in $\mathbb{R}^d$ ($d \sim 64$–$128$) with offset to keep all features in the positive orthant. Activations:

$$x = c_0 + \sum_j \alpha_j(\theta) \cdot u_j + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_d)$$

where $\{u_j\}$ span the embedding subspace and $\alpha_j(\theta)$ are the manifold-parameterizing coordinates (e.g., $\cos\theta, \sin\theta$ for $S^1$). Noise $\sigma \approx 0.05$–$0.1 \times \|f\|$.

**Critical:** the underlying $M$ must be densely sampled (continuous parameter, not a discrete set), otherwise any SAE trivially recovers a degenerate T1 representation that contains no information about $T$.

### Metrics per cell

1. **Live-atom count.** Atoms firing on $> 10^{-3}$ of inputs (threshold to be fixed and reported).
2. **T1 PH.** Persistence diagram of the live-atom point cloud $\{d_i : i \text{ live}\}$. Report $H_0$, $H_1$ persistence; bottleneck distance to ground-truth diagram.
3. **T2 PH.** Persistence diagram of the code point cloud projected onto top-$k$ PCs (start with $k = 4$, validate). Same persistence statistics.
4. **Reconstruction loss.** To rule out "the SAE just didn't learn."
5. **T1/T2 score.** Quantitative continuum, not binary. Candidate definition: ratio of normalized $H_1$ persistences, $\rho_{T1} / (\rho_{T1} + \rho_{T2})$. To be designed and validated during Stage 0.

### Calibration diagnostic (run first, before any architecture comparison)

ReLU + L1 only, on dense $S^1$ data, sweep $m \in \{16, 64, 256, 1024\}$. For each width, plot:

- Live-atom count vs. $m$
- T1 $H_1$ persistence vs. $m$
- T2 $H_1$ persistence vs. $m$
- Reconstruction loss vs. $m$

**Predicted outcome:** ReLU + L1 produces smoothly increasing live-atom count, T1 $H_1$ signature detectable once live atoms exceed $\sim 30$, essentially absent T2 signature throughout. If this prediction fails, the framework needs revisiting before architecture comparisons.

## Implementation stages

### Stage 0 — Infrastructure

- PH pipeline using `ripser` (faster than `gudhi` for VR up to $H_1$). Validate on point clouds with known homology: line segment, $S^1$, $S^2$, two disjoint circles, figure-8.
- Decide metric: Euclidean as default; cosine as comparison. Document choice.
- Decide on filtration: Vietoris-Rips. Skip alpha/witness filtrations.
- Implement bottleneck distance computation between persistence diagrams.

### Stage 1 — DGP

- Single function/class taking topology spec (`points`, `circle`, `two_circles`, `figure_eight`) and parameters $(d, \sigma, c_0, N)$, producing activations.
- Visualize generated activations projected onto the embedding subspace; sanity check.
- Compute and store ground-truth PH for each topology at the relevant sample density.

### Stage 2 — SAE training

- Minimal ReLU + L1 SAE: encoder $W_{\text{enc}} \in \mathbb{R}^{m \times d}$, decoder $W_{\text{dec}} \in \mathbb{R}^{d \times m}$, encoder bias, L1 penalty on activations.
- Standard training tricks: dead-atom resampling or ghost gradients, decoder normalization.
- Train on $S^1$ DGP at $m = 64$. Verify reconstruction, sparsity, that decoder atoms approximately tile the circle.
- Compute and inspect T1 and T2 PH on this single trained SAE.

### Stage 3 — Calibration diagnostic

Run the width sweep specified above. Decide whether to proceed based on whether predictions hold.

### Stage 4 — Architecture sweep

- Add TopK (with $k$ as architecture hyperparameter; pick $k = 2$ or $k = 4$ as default).
- Run full factorial: 4 topologies × 2 architectures × 3–4 widths × 3 seeds. Roughly 72–96 runs total.
- Each run: train SAE, compute T1 PH, T2 PH, live-atom count, reconstruction loss, T1/T2 score.
- Aggregate into results table.

### Stage 5 — Analysis & writeup

- Cross-architecture comparison via bottleneck distance on PH at matched cells.
- Identify regimes where the T1/T2 prediction holds vs. fails.
- Targeted ablations: noise $\sigma$, ambient $d$, L1 coefficient or TopK $k$, on the most discriminative cells.

### Future work (not in class report scope)

- JumpReLU, SpaDE.
- 2-manifold topologies ($T^2$, $S^2$).
- Real LLM activations (requires a method for establishing ground-truth topology of features in real models — open problem).
- Topology evolution during training on tasks with known structural transitions (e.g., modular arithmetic, grokking).

## Key design decisions and risks

### Decisions to make explicitly

- **PH metric.** Euclidean vs. cosine on decoder atoms. Likely Euclidean default.
- **PCA dimension for T2.** Start with 4, validate that varying this within reasonable range (2–8) doesn't qualitatively change results.
- **Live-atom threshold.** Suggested $10^{-3}$ activation rate; fix and report.
- **Decoder normalization.** Standard practice is to constrain $\|d_i\| = 1$; do this.
- **Magnitude vs. direction in PH.** With normalized decoders, this collapses; report.

### Known risks

- **T2 measurement is methodologically rickety.** PH of high-dimensional sparse codes is not standard; PCA reduction is a hyperparameter; results may be sensitive. Validate carefully on cases with known answers before trusting cross-architecture comparisons.
- **T1/T2 may not split cleanly across architectures.** Both may default to T1 in overcomplete regimes; the architectural-bias story may have to retreat to "T1/T2 differ only under width constraints," which is a weaker claim.
- **Figure-8 PH is geometrically subtle.** The wedge point causes finite-sample artifacts; ground-truth PH itself depends on sampling density near the wedge. Test carefully or drop in favor of two disjoint circles plus a separate connectivity test.
- **SAE training variance.** Real and large; use multiple seeds, do not trust single runs.
- **Fallback narrative.** If the headline T1/T2 prediction fails messily, the contribution becomes "we introduce a measurement framework for distinguishing T1 vs. T2 SAE representations and characterize its behavior on toy topologies." Methods contribution rather than findings contribution; still publishable.

## Key references

- Hindupur, S.S., Lubana, E.S., Fel, T., Ba, D. (2025). *Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry*. arXiv:2503.01822.
- Engels, J., et al. (2024). *Not All Language Model Features Are Linear*.

## Repository conventions (suggested)

```
sae_topology/
  dgp/              # data generating processes per topology
  saes/             # SAE architectures (relu, topk, ...)
  ph/               # persistent homology utilities
  experiments/      # runnable experiment scripts
  analysis/         # plotting, aggregation, table generation
  configs/          # YAML configs per experiment
  results/          # logged outputs, persistence diagrams, checkpoints
```

Each experiment script consumes a config and writes results to a timestamped subdirectory of `results/`. Persistence diagrams stored as `.npy` (birth/death pairs); SAE checkpoints as `.pt`.
