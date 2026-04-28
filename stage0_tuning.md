# Stage 0: Pipeline Validation and Tuning

**Purpose:** Establish that the Mapper + Laplacian topology-detection pipeline correctly recovers known manifold structure on raw samples, before introducing any SAE. Stage 0 is a *necessary condition* for trusting any downstream result: if the pipeline cannot recover $T^2$'s topology from clean torus samples, no claim about the topology of SAE post-activations is interpretable.

This document specifies (a) what to tune, (b) what success looks like, (c) what to freeze and carry forward to Stages 1–3, and (d) what to recompute per-dataset rather than freeze.

---

## 1. Scope and Position in the Pipeline

```
Stage 0  →  Stage 1  →  Stage 2  →  Stage 3
[VALIDATE]  [TRAIN]    [ANALYZE]   [AGGREGATE]
   ↑
   YOU ARE HERE
```

Stage 0 runs the full Mapper + Laplacian pipeline on **raw uniform samples** drawn directly from each target manifold (no SAE involved). Outputs of Stage 0 are:

1. A validated configuration (sample size, kNN parameters, Mapper grid bounds) per manifold.
2. Empirical confirmation that both Mapper and Laplacian methods recover the correct topology and spectrum on clean data.
3. A noise-perturbation robustness check (§5) confirming the pipeline handles perturbations of the kind SAEs will introduce.

If Stage 0 fails for a given manifold, **do not proceed to Stage 1 for that manifold.** Either tune further, increase sample size, or remove the manifold from the experimental set.

---

## 2. Manifolds Covered in Stage 0

| Manifold | Dim | Ambient (raw) | Closed-form spectrum reference |
|----------|-----|----------------|-------------------------------|
| $S^1$ | 1 | $\mathbb{R}^2$: $(\cos\theta, \sin\theta)$ | $\lambda_k = k^2$, mult 2 for $k>0$ |
| $T^2$ | 2 | $\mathbb{R}^4$: $(\cos\theta, \sin\theta, \cos\phi, \sin\phi)$ | $\lambda_{m,n} = m^2 + n^2$, mult $r_2(\lambda)$ |
| $S^2$ | 2 | $\mathbb{R}^3$: unit sphere | $\lambda_\ell = \ell(\ell+1)$, mult $2\ell+1$ |

**Note:** Stage 0 uses each manifold's natural low-dimensional embedding, **not** the higher-dimensional ambient $\mathbb{R}^D$ used in Stage 1. The Stage-0 pipeline is validated on the low-dim embedding; Stage 1 will lift samples to $\mathbb{R}^D$ before feeding them through the SAE, and post-activation analysis happens in $\mathbb{R}^m$ (SAE latent space). The validated configuration transfers across these spaces only along the dimensions noted in §6.

---

## 3. Tunable Parameters

Listed in the order they should be tuned. Earlier parameters constrain the search space for later ones.

### 3.1 Sample size $N$

The single most consequential parameter. Too small: graph Laplacian eigenvalues are noisy, Mapper cover-cells are sparse, no stable region. Too large: kNN graph construction and eigendecomposition become slow, but compute is parallelized on a remote cluster so this is not a binding constraint at the target scale.

**Target:** $N = 100{,}000$ for $S^1$, $T^2$, and $S^2$.

**Sweep range (validation):** $N \in \{10{,}000, 25{,}000, 50{,}000, 100{,}000\}$.

**Selection rule:** confirm validation criteria (§4) pass at $N = 100{,}000$, then verify they continue to pass (with comparable margins) at $N = 50{,}000$ and $N = 25{,}000$. The point is not to find the smallest workable $N$ — Stage 2 will use the same $N = 100{,}000$ as Stage 0 — but to confirm the result is not an artifact of one specific sample size. If validation degrades sharply between 50K and 100K, the pipeline is undersampled at 100K and the manifold needs more samples, not less.

### 3.2 kNN parameter $k$

Controls the locality of the graph. Too small → graph disconnects, spurious connected components, low eigenvalues clump near zero. Too large → cross-manifold shortcuts (e.g., kNN connects opposite sides of the torus), eigenvalue spectrum smeared, geometric structure lost.

**Sweep range:** $k \in \{15, 25, 40\}$.

**Selection criterion:** the $k$ that produces the most stable Laplacian spectrum — specifically, the smallest log-ratio error $\mathcal{E}$ (defined in §4.2) on raw samples.

### 3.3 Kernel bandwidth $\sigma$

Sets the Gaussian weight scale on graph edges. Use the **median heuristic**: for each dataset, compute the median pairwise distance among kNN edges and set $\sigma$ equal to that median. This is the standard choice, robust across scales.

**Do not freeze a numerical value of $\sigma$.** Freeze the *rule* (median heuristic). $\sigma$ is recomputed per dataset (different SAEs produce different post-activation densities).

### 3.4 Mapper hyperparameter grid

Specifies the parameter sweep used by persistent Mapper to identify stable Betti regions.

**Default grid:**

| Parameter | Values |
|-----------|--------|
| n_intervals (per filter axis) | $\{5, 8, 12, 18, 25, 35\}$ |
| overlap (fractional) | $\{0.15, 0.25, 0.35, 0.50\}$ |
| Clustering | single-linkage with first-gap dendrogram threshold (fixed) |

**Total configurations per (manifold, filter):** $6 \times 4 = 24$.

**Why uniform across filter dimensions.** With $N = 100{,}000$ samples and the manifold occupying a $d$-dimensional submanifold of the $k$-dimensional filter image, points-per-non-empty-cover-box $\approx N / n^d$ (where $n$ is n_intervals). For the most demanding case (4-D Laplacian filter on $T^2$, which is 2-D) at $n = 35$: $\sim 35^2 = 1225$ non-empty boxes, $\sim 80$ points/box — well above the $\sim 20$ minimum for stable single-linkage clustering. Lower-dimensional filters get more points/box at the same $n$. Uniform grid is therefore safe at the target $N$.

**Validation requirement:** the largest stable Betti region must be **interior** to the grid, i.e., not touching either the n_intervals or overlap edge. If it does touch an edge, expand the grid in that direction (e.g., add n_intervals = 50, or overlap = 0.60) and re-sweep. The grid you carry forward to Stage 2 must contain the Stage-0 stable region with margin.

### 3.5 Number of Laplacian eigenvalues $K$

How many eigenvalues to compute and compare to the closed-form spectrum.

**Default:** $K = 20$. Sufficient to detect multiplicity patterns up to a useful depth without inflating eigendecomposition cost. Do not change without specific reason.

### 3.6 Multiplicity-cluster tolerance $\epsilon$

For the multiplicity preservation check (§4.2). Empirical eigenvalues at the same theoretical level won't match exactly; cluster them with tolerance $\epsilon$.

**Default:** $\epsilon = 0.05$ (in normalized-ratio units). Tune up to 0.10 if the spectrum is noisier than expected; tune down if multiplicity clusters are clearly separated.

---

## 4. Validation Criteria

A manifold passes Stage 0 only when **all** of the following hold for raw samples.

### 4.1 Mapper criteria

Run the persistent Mapper sweep using the **Laplacian eigenvector filter** — the same filter that will serve as Stage 2's primary metric. The point of Stage 0 is to validate the actual primary metric, not a different (and easier) variant. GT-filter Mapper is *not* validated at Stage 0; the Mapper machinery itself (cover construction, single-linkage clustering, nerve-edge logic) is well-established and does not require empirical sanity-checking on each manifold.

**Filter dimensionality:**

| Manifold | Filter dim | First non-trivial LB eigenfunctions |
|----------|------------|-------------------------------------|
| $S^1$ | 2 | $\cos\theta, \sin\theta$ (eigenvalue 1, mult 2) |
| $T^2$ | 4 | $\cos\theta, \sin\theta, \cos\phi, \sin\phi$ (eigenvalue 1, mult 4) |
| $S^2$ | 3 | degree-1 spherical harmonics = $x, y, z$ on $S^2$ (eigenvalue 2, mult 3) |

**Note on eigenvalue degeneracy.** For $T^2$ and $S^2$, the first $k$ non-trivial eigenvalues are degenerate (mult 4 and 3 respectively). The numerical eigensolver returns *some* basis of the corresponding eigenspace, not necessarily the natural Cartesian-coordinate basis. The Mapper graph is invariant under orthogonal rotations of the filter image (axis-aligned cover boxes in a rotated basis still cover the same manifold with the same nerve structure up to relabeling), so the Betti output should be unaffected. Verify this empirically: the result should be reproducible across random seeds even though the eigenvector basis varies.

Three conditions:

1. **Correct Betti.** The largest stable Betti region reports the correct $(b_0, b_1)$ for the manifold:
   - $S^1$: $(1, 1)$
   - $T^2$: $(1, 2)$
   - $S^2$: $(1, 0)$
2. **Substantial stable region.** The stable region must contain at least **6 of 24 grid configurations** (≥25%) with the correct Betti, forming a *contiguous* block in (n_intervals, overlap) space.
3. **Sensible failure modes outside the stable region.** Configurations with Betti differing from the stable value should fail in interpretable directions: coarser covers (low n_intervals) should give $b_1 < $ correct (cycles missed); finer covers (high n_intervals) should give $b_1 > $ correct (spurious cycles from sampling noise). If failures are erratic — e.g., $b_1 = 5$ at coarse and $b_1 = 0$ at fine — the pipeline is broken even when the stable region looks correct.

### 4.2 Laplacian criteria

Build the Coifman–Lafon $\alpha=1$ normalized graph Laplacian on raw samples (procedure in §4.4). Compute first $K = 20$ eigenvalues. Three conditions:

1. **Log-ratio error.** Define
$$\mathcal{E} = \frac{1}{K-1} \sum_{i=2}^{K} \left| \log(\hat\lambda_i / \hat\lambda_1) - \log(\lambda_i / \lambda_1) \right|$$
where $\hat\lambda_i$ are empirical and $\lambda_i$ are closed-form. Require $\mathcal{E} < 0.05$.
2. **Multiplicity preservation.** For each theoretical eigenvalue level (in ratio), count empirical eigenvalues within $\pm \epsilon$ of that level and compare to theoretical multiplicity. Expected patterns:
   - $S^1$: pairs at ratios 1, 4, 9, 16, ...
   - $T^2$: clusters of size 4 at ratios 1, 2, 4; cluster of size 8 at ratio 5; gaps at 3, 6, 7
   - $S^2$: clusters of size 3, 5, 7, 9, ... at ratios 1, 3, 6, 10, ... (after dividing by $\lambda_1 = 2$)
   The first 3–4 multiplicity clusters must match exactly; later clusters may show $\pm 1$ count error due to numerical noise.
3. **Component count.** Number of near-zero eigenvalues (below $0.1 \cdot \hat\lambda_1$) equals $b_0$ of the manifold (1 for all single-component manifolds in Stage 0).

### 4.3 Coifman–Lafon Laplacian construction (reference)

For samples $\{x_i\}_{i=1}^N$:

1. Build kNN graph (parameter $k$ from §3.2).
2. Compute pairwise distances on kNN edges; set $\sigma$ = median of these distances.
3. Form weights $W_{ij} = \exp(-\|x_i - x_j\|^2 / (2\sigma^2))$ on kNN edges; symmetrize.
4. Density correction: compute degree $D_{ii} = \sum_j W_{ij}$ and reweight $\tilde W = D^{-1} W D^{-1}$.
5. Random-walk normalization: compute $\tilde D_{ii} = \sum_j \tilde W_{ij}$ and form $L = I - \tilde D^{-1} \tilde W$.
6. Compute first $K$ eigenvalues via sparse eigensolver (e.g., `scipy.sparse.linalg.eigsh` with `sigma=0`, `which='LM'`).

---

## 5. Robustness Check (Required)

Validation on perfectly clean samples is *necessary but not sufficient*. The pipeline must also be robust to perturbations of the kind SAEs will introduce. Without this check, a pipeline that works at the edge of its operating regime will silently fail on SAE outputs even when the SAE preserves topology.

**Procedure:**

1. Take the validated configuration ($N$, $k$, Mapper grid) for a manifold.
2. Generate raw samples and add isotropic Gaussian noise of magnitude $\eta \cdot \text{(intrinsic diameter)}$, with $\eta \in \{0.01, 0.05, 0.10\}$.
3. Re-run the full pipeline (Mapper + Laplacian).
4. Verify graceful degradation:
   - Mapper Betti remains correct at $\eta = 0.01$ and $\eta = 0.05$.
   - Laplacian $\mathcal{E}$ remains under 0.10 at $\eta = 0.05$.
   - At $\eta = 0.10$, mild degradation is acceptable (Mapper Betti may drift, $\mathcal{E}$ may approach 0.20) — but the result should still be interpretable, not chaotic.

If perturbation breaks the pipeline at $\eta = 0.05$, the validated configuration is operating at the edge of its regime. Increase $N$ and re-validate.

The robustness check must pass for every manifold before that manifold proceeds to Stage 1.

---

## 6. What Carries Forward to Stages 1–3

This is the most important section. The transfer rules are nuanced because the parameters interact differently with raw samples (Stage 0) vs. SAE post-activations (Stage 2).

### 6.1 Frozen parameters (use Stage-0 values directly in Stage 2)

| Parameter | Reason |
|-----------|--------|
| Sample size $N$ | Stage 2 uses $N_{\text{eval}} = 100{,}000$ fresh samples per (architecture, manifold), matching Stage 0 exactly. Stage 1 trains on a separate $N_{\text{train}} = 1{,}000{,}000$ samples (10×). Compute is parallelized across cluster, so equal-$N$ Stage 0 and Stage 2 is the cleaner methodological choice and is affordable. |
| Mapper hyperparameter grid bounds | The grid $\{5, 8, 12, 18, 25, 35\} \times \{0.15, 0.25, 0.35, 0.50\}$ (or whatever expanded grid Stage 0 settled on) is reused in Stage 2. **Each Stage-2 case re-sweeps the grid; do not freeze the specific winning config.** |
| Number of Laplacian eigenvalues $K$ | Use $K = 20$ throughout. |
| Multiplicity tolerance $\epsilon$ | Use Stage-0 value (default 0.05) throughout. |
| Coifman–Lafon $\alpha=1$ normalization | Use throughout. |
| Single-linkage first-gap clustering for Mapper | Use throughout. |
| Median-heuristic rule for $\sigma$ | The *rule*, not the value. |

### 6.2 Recomputed per-dataset (not frozen)

| Parameter | Reason |
|-----------|--------|
| Kernel bandwidth $\sigma$ | Recompute via median heuristic on each dataset. SAE post-activations have different point density than raw samples; the Stage-0 numerical $\sigma$ is wrong for them. |
| Specific (n_intervals, overlap) winning config | Each Stage-2 case sweeps the full grid and reports its own stable region. The Stage-0 winner is not necessarily the Stage-2 winner because post-activation point density differs from raw density. |
| kNN $k$ | Re-validate from $\{15, 25, 40\}$ on each new dataset by selecting the value with smallest $\mathcal{E}$. The Stage-0 winner is a strong prior but not authoritative. |

### 6.3 Critical pitfall to avoid

**Do not pick a single (n_intervals, overlap) that worked on raw samples and apply it to SAE post-activations.** The two point clouds have different ambient spaces, different intrinsic scales, and potentially very different densities. A cover with n_intervals=12 on raw $T^2$ samples in $\mathbb{R}^4$ may produce vastly different cell sizes (in absolute terms) on post-activations in $\mathbb{R}^m$. Letting each case find its own stable region via the full sweep is what makes the cross-condition comparison honest.

The role of Stage 0 is to prove the pipeline *can* recover topology when topology is present and to define the parameter ranges within which we'll search — not to fix a single config that we then apply universally.

### 6.4 Summary of carry-forward

```
STAGE 0 OUTPUT:
  per-manifold:
    - N = 100,000 (validated; frozen for Stage 2)
    - Mapper grid bounds {5,8,12,18,25,35} × {0.15,0.25,0.35,0.50}
      (frozen; expand if needed during validation)
    - Working (n_intervals, overlap) on raw samples (NOT frozen — anchor only)
    - Working k on raw samples (NOT frozen — strong prior)
    - σ rule: median heuristic (frozen rule)

STAGE 2 USE:
  for each (architecture, manifold):
    - Use N_eval = 100,000 (matches Stage 0)
    - Sweep Mapper grid bounds → find this case's stable region
    - Sweep k ∈ {15, 25, 40} → pick most stable
    - Compute σ via median heuristic on this case's data
    - Report Betti from this case's stable region (not Stage-0's)
```

---

## 7. Implementation Order

**Do not attempt to validate all manifolds in parallel.** The pipeline is most usefully built and debugged on the simplest case first.

1. **$S^1$ first.**
   - Implement raw-sample generator.
   - Implement Coifman–Lafon Laplacian and verify $\mathcal{E} < 0.05$ on $S^1$ before touching Mapper.
   - Implement Mapper with GT filter; verify stable $(1, 1)$ region.
   - Run robustness check (§5). Fix any failures.
   - Lock in the $S^1$ Stage-0 configuration.
2. **$T^2$ second.** Same procedure. Most likely to need the largest $N$.
3. **$S^2$ third.** Same procedure.

If any step fails after thorough tuning (e.g., $T^2$ does not validate at $N = 100{,}000$), document the failure, do not lower the validation threshold, and consider whether to drop that manifold from the experimental set or to invest in larger $N$.

---

## 8. Output Artifacts

Stage 0 produces a **validation report** (one per manifold) containing:

1. Final validated $N$, $k$, Mapper grid bounds.
2. Mapper sweep result table: all 24 (or expanded) configurations, each with computed $(b_0, b_1)$, node count, edge count.
3. Stable region identification: which contiguous block of configurations agrees on Betti, and what value.
4. Laplacian results: first 20 eigenvalues (raw and ratios), $\mathcal{E}$, multiplicity-cluster check, near-zero count.
5. Robustness-check results: Mapper Betti and Laplacian $\mathcal{E}$ at $\eta \in \{0.01, 0.05, 0.10\}$.
6. Plot: persistent Mapper grid heatmap showing $(b_0, b_1)$ at each (n_intervals, overlap) configuration. Stable region should be visually contiguous.
7. Plot: empirical vs. closed-form eigenvalue ratios (first 20), with multiplicity clusters annotated.

In addition, Stage 0 must **persist the validated Mapper graphs themselves** (not just summary statistics) for use as reference graphs in Stage 2 comparisons:

8. **Reference Mapper graphs** — for every configuration in the stable region, save the full Mapper graph (nodes with their underlying point indices; edges; node-position embeddings if computed). Serialize in a format that supports later structural comparison (e.g., graph edit distance, spectral graph distance, Betti decomposition by connected component). Recommended: NetworkX pickle or GraphML, with a sidecar JSON for metadata (n_intervals, overlap, $k$, filter type, manifold, sample seed).
9. **Reference Laplacian eigenvectors** — save the first $k$ non-trivial eigenvectors (those used as the filter) along with their eigenvalues and the random seed. These are needed to interpret Stage 2 Mapper graphs that use the Laplacian filter, since the eigenvector basis is rotation-ambiguous within degenerate eigenspaces.
10. **Sample point clouds** — save the raw sample arrays $\{x_i\}$ that produced the validated graphs, along with the random seed. Stage 2 will not reuse these samples (it uses fresh SAE post-activation point clouds), but having the exact Stage-0 cloud available makes any future re-validation or baseline regeneration deterministic.

These artifacts are not just sanity-check material. They are the **ground-truth comparison set** for Stage 2: when Stage 2 produces a Mapper graph from SAE post-activations on $T^2$, the meaningful question is "how does this graph compare to the Stage-0 reference graph for $T^2$?" — both at the level of Betti numbers and at the level of structural features (node count, edge count, nerve match, node-to-box ratio). Without persisted reference graphs, Stage 2 can only compare against scalar summaries, losing the structural signal that motivates using Mapper in the first place.

These artifacts also become reference data for interpreting Stage 2 quantitative results — e.g., when Stage 2 reports $\mathcal{E} = 0.30$ on some SAE post-activation, the Stage-0 $\mathcal{E} = 0.03$ on raw samples gives the baseline for what "good" looks like.

---

## 9. Failure Modes and Diagnostics

If Stage 0 fails for a manifold, diagnose in this order:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Mapper stable region too small (<25%) | $N$ too small for cover density | Increase $N$ |
| Mapper stable region at grid edge | Grid too narrow | Expand grid |
| Mapper Betti correct but eigenvalue $\mathcal{E}$ high | $k$ wrong, or kernel bandwidth issue | Sweep $k$, verify median heuristic |
| Multiplicity check fails but ratios match | $\epsilon$ too tight or too loose | Tune $\epsilon$ in [0.03, 0.10] |
| Spurious near-zero eigenvalues | $k$ too small (graph disconnecting) | Increase $k$ |
| Eigenvalues smeared, multiplicities lost | $k$ too large (shortcuts) | Decrease $k$ |
| Robustness fails at $\eta = 0.05$ but clean samples pass | Pipeline at edge of operating regime | Increase $N$ |
| Mapper failure modes erratic (not monotone in n_intervals) | Pipeline bug or unstable clustering | Re-examine clustering threshold; check for ties in dendrogram |

If none of these fixes resolve the failure, the manifold likely cannot be validated with the current pipeline at reasonable $N$. Document and exclude from the experimental set, or invest in larger $N$ at significant compute cost.

---

## 10. Sign-off Checklist

Before declaring Stage 0 complete for a manifold:

- [ ] Mapper stable region (under **Laplacian eigenvector filter**) contains correct Betti at $N = 100{,}000$
- [ ] Stable region size ≥ 6 of 24 grid configurations
- [ ] Stable region is contiguous in parameter space
- [ ] Stable region is interior to grid (not at edge of either n_intervals or overlap)
- [ ] Mapper failure modes outside stable region are interpretable (monotone in n_intervals)
- [ ] Validation continues to pass (with comparable margins) at $N = 50{,}000$ and $N = 25{,}000$ — confirms result is not artifact of one specific sample size
- [ ] Laplacian $\mathcal{E} < 0.05$ on raw samples at $N = 100{,}000$
- [ ] Multiplicity-cluster check passes for first 3–4 clusters
- [ ] Near-zero eigenvalue count = $b_0$
- [ ] Robustness check passes at $\eta = 0.05$
- [ ] Eigenvector-basis-rotation reproducibility check passes for $T^2$ and $S^2$ (Mapper Betti unchanged across random seeds despite eigenvector basis varying within degenerate eigenspaces)
- [ ] Validation report (§8) generated and saved
- [ ] Reference Mapper graphs persisted for every configuration in stable region (for Stage 2 comparison)
- [ ] Reference Laplacian eigenvectors and seeds persisted
- [ ] Sample point clouds and seeds persisted
- [ ] Frozen parameters (§6.1) recorded for use in Stage 2
