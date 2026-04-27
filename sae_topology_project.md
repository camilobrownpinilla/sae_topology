# Topological Inductive Biases of Sparse Autoencoders

**Author:** Camilo Brown-Pinilla
**Course:** AM231
**Status:** Project specification (pre-implementation)

---

## 1. Research Question

> Is the topology an SAE can recover predictable from its architecture?

Concretely: when an SAE encodes inputs sampled from a known manifold $M$, does the topology of $M$ survive the encoding, and does survival depend systematically on the SAE's nonlinearity / loss / sparsity mechanism?

This builds on **"Projecting assumptions"** (the inductive-bias-as-projection framing), which establishes that different SAE architectures define different sets of recoverable features. We extend this from feature *identity* to feature-manifold *topology*, motivated by mounting evidence (Engels et al., "Not all language model features are linear"; circular features for days/months; board-state manifolds) that real LM features are not always 1-D directions.

We work entirely with **synthetic toy models** where ground-truth manifold structure is known. Real-LM features are out of scope.

---

## 2. What "Recovery" Means

### 2.1 Setup

An SAE is parameterized as
$$z(x) = \sigma(W_e x + b_e), \qquad \hat{x}(x) = W_d \, z(x) + b_d$$
trained to minimize $\sum_x \|x - \hat{x}\|_2^2 + \lambda R(z)$. Four objects are available to probe:

1. **Pre-activations** $z_{\text{pre}}(x) = W_e x + b_e \in \mathbb{R}^m$
2. **Post-activations** $z(x) = \sigma(z_{\text{pre}}(x)) \in \mathbb{R}^m_{\geq 0}$
3. **Reconstructions** $\hat{x}(x) \in \mathbb{R}^d$
4. **Dictionary** $W_d \in \mathbb{R}^{d \times m}$ (columns = feature directions)

### 2.2 Why post-activations are the primary object

The dictionary alone is topologically uninformative. Consider the unit circle in $\mathbb{R}^d$. Under standard ReLU-like nonlinearities, post-activations are non-negative, so the minimum dictionary capable of representing the full circle has **four** atoms (a "+/− cross": $\pm u, \pm v$ spanning the circle's plane), not two — two would only cover the positive quadrant. The 4-atom dictionary is the most efficient possible representation, but visually it is just four vectors in a cross pattern; the dictionary alone reveals nothing about the circle's topology. The same circle could alternatively be represented as an $m$-tiling (many atoms arranged around the circle); here the dictionary visually traces the circle, but this is a memorization-style representation, less efficient than the cross.

By contrast, post-activations $z(x_i)$ for $\{x_i\}$ densely sampled from the circle preserve the topology in **both** representations:
- *Cross representation:* $z(x) \in \mathbb{R}^4_{\geq 0}$ has support pattern that rotates continuously as $x$ moves around the circle, tracing a closed loop.
- *Tiling representation:* support of $z(x)$ shifts continuously through the (arbitrarily indexed) feature dimensions, again tracing a loop in $\mathbb{R}^m_{\geq 0}$.

Hence:

> **The SAE recovers the topology of $M$ if a kNN graph on $\{z(x_i) : x_i \sim M\}$ has Betti numbers matching those of $M$. Recovery is metrically faithful if additionally the graph Laplacian spectrum matches the Laplace–Beltrami spectrum of $M$ in ratio.**

This separates **topological fidelity** (Mapper / Betti) from **metric fidelity** (spectrum) — a distinction Section 4 elaborates.

### 2.3 Secondary objects

- **Pre-activations:** baseline. A linear map $W_e$ generically preserves topology unless degenerate. If pre-activations recover $M$ but post-activations don't, the topology-destroying mechanism is localized to $\sigma$.
- **Reconstructions:** diagnostic. Tests whether $M$ lies in the image of the SAE as a function. If $\hat{x}$ recovers $M$ but $z$ doesn't, the decoder is gluing things back together that the latent code has separated.
- **Dictionary:** not directly probed (see argument above).

---

## 3. Manifolds

All manifolds sampled uniformly with respect to natural Riemannian volume.

### 3.1 Smooth manifolds (full pipeline applies)

| Manifold | Dim | Embedding | Closed-form LB spectrum |
|----------|-----|-----------|-------------------------|
| $S^1$ (circle) | 1 | $(\cos\theta, \sin\theta) \in \mathbb{R}^2$ | $\lambda_k = k^2$, mult 2 for $k>0$ |
| $T^2$ (flat torus) | 2 | $\mathbb{R}^2 / 2\pi\mathbb{Z}^2$, embedded in $\mathbb{R}^4$ as $(\cos\theta, \sin\theta, \cos\phi, \sin\phi)$ | $\lambda_{m,n} = m^2 + n^2$, mult $r_2(\lambda)$ |
| $S^2$ (sphere) | 2 | Unit sphere in $\mathbb{R}^3$ | $\lambda_\ell = \ell(\ell+1)$, mult $2\ell+1$ |

**Sphere sampling:** draw $x \sim \mathcal{N}(0, I_3)$, normalize to unit length. Yields uniform samples on $S^2$.

**Sphere ground-truth filter:** use the 3D Cartesian coordinates $(x, y, z)$ directly — injective on $S^2$, avoiding the polar singularities of spherical coordinates. The first three non-trivial LB eigenfunctions of $S^2$ are degree-1 spherical harmonics, which equal $x, y, z$ restricted to the sphere; the Laplacian eigenvector filter should therefore recover the Cartesian embedding up to orthogonal rotation.

**Embedding into ambient SAE input space:** lift each manifold's natural embedding into ambient $\mathbb{R}^D$ (SAE input dimension, $D \in [64, 256]$) by appending zeros + small isotropic Gaussian noise ($\sigma_{\text{noise}} = 0.01$). Simulates the manifold being embedded in a higher-dimensional residual-stream-shaped ambient.

### 3.2 Singular space (Mapper-only)

| Space | Note |
|-------|------|
| Figure-8 | Not a manifold — singular vertex where two loops meet. Laplace–Beltrami does not apply. **Use Mapper only** for figure-8; spectral analysis is skipped. |

---

## 4. Methods: Topology vs. Geometry

The two main tools answer different questions. **Report both as separate columns in results.**

| Tool | Question answered | Output |
|------|-------------------|--------|
| Mapper (Laplacian eigenvector filter primary) | Is the topological type preserved? | $b_0, b_1$ of Mapper graph; diagnostics below |
| Graph Laplacian spectrum | Is the metric shape preserved? | Eigenvalue ratio error; multiplicity pattern |

A "topology-preserved, metric-distorted" finding (correct Betti, large spectral error) is itself an interesting architectural result, not a method failure.

### 4.1 Mapper

**Pipeline:**

1. Compute filter $f: \mathbb{R}^m \to \mathbb{R}^k$ on post-activations.
2. Cover $f(Z) \subset \mathbb{R}^k$ with overlapping axis-aligned boxes (n_intervals per axis, fractional overlap).
3. For each box $U$, cluster $\{z_i : f(z_i) \in U\}$ in **post-activation space** using single-linkage with first-gap threshold on the dendrogram. Each cluster → node.
4. Add edge between nodes whose underlying point sets share at least one point.
5. Compute $b_0$ (components) and $b_1 = |E| - |V| + b_0$.

**Filter choices (run all three; primary vs. diagnostic distinction):**

- **Laplacian eigenvector filter (primary).** First $k$ non-trivial eigenvectors of the kNN Laplacian on $Z$ (typically $k=3$ for $S^1$, $k=4$ for $T^2$ and $S^2$). Derives the cover from the data, so genuinely tests whether the SAE preserves enough geometric structure for data-driven coordinates to rediscover the manifold. **This is the methodologically honest "did the SAE work?" question.**
- **Ground-truth filter (diagnostic / upper bound).** Intrinsic coordinates of the source manifold ($(\theta, \phi)$ for $T^2$; $(x, y, z)$ for $S^2$). Available only because of the toy-model setup. Mapper under this filter measures local connectivity preservation *given the optimal cover*; it cannot honestly serve as the primary recovery metric (see below). The pair (GT-pass, Laplacian-fail) is itself an interesting failure mode: "topology preserved in a basis the data-driven method can't find."
- **PCA filter (linear baseline).** Top-$k$ principal components of $Z$. Tests whether linear methods suffice; expected to fail when the activation embedding is curved.

**Why GT filter cannot be the primary metric.** The cover comes from the source coordinates, not the data, so the cover combinatorics get baked into the result regardless of what the SAE did. In the limit where the SAE collapses all of $T^2$ to a single point $z^* \in \mathbb{R}^m$: the GT-filter cover still tiles $[0, 2\pi)^2$ as a toroidal grid, every preimage equals $\{z^*\}$ (clustering trivially gives one node per box), overlapping cover boxes share source samples so edges fire between adjacent grid cells, and Mapper outputs the toroidal grid graph with $b_1 = 2$. Topology "recovered" despite total collapse. The Laplacian eigenvector filter avoids this trap — collapse produces degenerate eigenvectors and a degenerate cover, so failure is honest. Spectral analysis catches the collapse independently, so the project-level pipeline isn't fooled even with GT-only Mapper, but any single tool we call primary should be honest in isolation.

**Hyperparameter sweep (persistent Mapper):**

| Parameter | Values |
|-----------|--------|
| n_intervals | $\{5, 8, 12, 18, 25\}$ |
| overlap | $\{0.15, 0.25, 0.35, 0.50\}$ |
| Clustering | single-linkage with first-gap threshold (fixed) |

Full sweep = 20 configurations per (architecture, manifold, filter). Report Betti stability across grid; headline result = Betti values in the largest stable region.

**Mapper-specific diagnostics:**

1. *Betti numbers* $(b_0, b_1)$ — headline topology under each filter. Primary report under Laplacian filter; GT-filter values reported alongside as an upper bound.
2. *Node-to-box ratio* (computed under both GT and Laplacian filters) — total Mapper nodes divided by number of cover boxes containing data. Ratio = 1 is perfect; ratio > 1 indicates fragmentation. Under GT filter, fragmentation directly counts patch-tearings in source coordinates. Under Laplacian filter, fragmentation indicates that data-driven coordinates fail to identify local connectivity.
3. *Nerve comparison* (primarily under Laplacian filter) — for a regular cover of the eigenvector image, the expected Mapper graph is the nerve of that cover. For $S^1$ a cycle, $T^2$ a toroidal grid, $S^2$ a spherical triangulation. Report Betti mismatch vs. this expected nerve. Under GT filter the comparison is sharper but vulnerable to the cover-bias issue noted above; under Laplacian filter it's noisier but methodologically honest. (Singh–Mémoli–Carlsson nerve-based reasoning.)

### 4.2 Laplace–Beltrami operator (intrinsic reference)

For a compact Riemannian manifold $(M, g)$ without boundary, the LB operator $\Delta_g$ is the unique generalization of the Euclidean Laplacian that depends only on intrinsic geometry. It is self-adjoint with discrete spectrum
$$0 = \lambda_0 \leq \lambda_1 \leq \lambda_2 \leq \ldots \to \infty.$$

Key facts:

- **$b_0$:** multiplicity of $\lambda_0 = 0$ = number of connected components.
- **Weyl's law:** $\lambda_k \sim C(M) \cdot k^{2/d}$ for large $k$. Growth rate reveals dimension.
- **Spectrum determines geometry** (almost — exotic isospectral exceptions exist, none relevant here). It does **not** depend only on topology: stretching the torus changes the spectrum.

### 4.3 Graph Laplacian (empirical)

On post-activation samples $\{z_i\}_{i=1}^N \subset \mathbb{R}^m$:

1. Build kNN graph with $k \in \{15, 25, 40\}$ (sweep; report primary at most spectrally stable $k$).
2. Compute Gaussian weights $W_{ij} = \exp(-\|z_i - z_j\|^2 / 2\sigma^2)$, with $\sigma$ = median pairwise distance among kNN edges.
3. Apply **Coifman–Lafon $\alpha=1$ normalization**:
   - $D_{ii} = \sum_j W_{ij}$; reweight $\tilde W = D^{-1} W D^{-1}$.
   - $\tilde D_{ii} = \sum_j \tilde W_{ij}$; form $L = I - \tilde D^{-1} \tilde W$.
4. Compute first $K = 20$ eigenvalues via sparse eigensolver.

**Why Gaussian weights:** the convergence theorems of the empirical graph Laplacian to $\Delta_g$ (Belkin–Niyogi 2008; Coifman–Lafon 2006; Singer 2006) require a smooth kernel. The unweighted (binary) kNN Laplacian converges to a different operator with no closed-form LB reference, forcing comparison against a noisier numerical reference instead of the exact spectrum.

**Why Coifman–Lafon:** the unnormalized graph Laplacian converges to $\Delta_g + (\nabla\rho/\rho)$-correction terms when sampling density $\rho$ is non-uniform. The $\alpha=1$ normalization cancels this correction, giving pure $\Delta_g$ regardless of sampling. Uniform toy-manifold sampling makes the correction small in expectation, but Coifman–Lafon is cheap insurance against finite-$N$ density variation.

**$k$ selection rationale:** $k$ too small → graph disconnects, spurious $b_0$, low eigenvalues cluster near zero. $k$ too large → cross-manifold shortcuts (e.g., opposite sides of the torus connected), high eigenvalues blurred, geometric info lost. For $N \in [2000, 10000]$ and target manifolds, $k \in [15, 40]$ is the reasonable range; sweep within it and report the most stable choice from Stage 0 validation.

### 4.4 Comparison metrics

Empirical eigenvalues have arbitrary scale ($N, \sigma$, embedding-dependent). Use scale-invariant comparisons.

**Primary metric — log-ratio error:**
$$\mathcal{E} = \frac{1}{K-1} \sum_{i=2}^{K} \left| \log(\hat\lambda_i / \hat\lambda_1) - \log(\lambda_i / \lambda_1) \right|$$
where $\lambda_i$ are the closed-form LB eigenvalues. Lower is better; $\mathcal{E} = 0$ is perfect recovery up to scale.

**Multiplicity preservation:** for each theoretical eigenvalue level (in ratio), count empirical eigenvalues within $\pm\epsilon$ of that level. Compare to theoretical multiplicity.
- $T^2$: expect 4-clusters near $\lambda_1, 2\lambda_1, 4\lambda_1$, 8-cluster near $5\lambda_1$, gaps at $3\lambda_1, 6\lambda_1, 7\lambda_1$ (sums of two squares).
- $S^2$: expect odd-multiplicity clusters at sizes 3, 5, 7, 9, 11, ... — sharp qualitative discriminator from $T^2$'s even multiplicities.

**Connected components:** count near-zero eigenvalues (below, e.g., $0.1 \hat\lambda_1$). Should equal $b_0$ of the manifold.

### 4.5 Reference spectra (closed form)

Ratios normalized by $\lambda_1$.

**$S^1$ (unit circle):** $\lambda_k = k^2$, mult 2 for $k > 0$.
Ratios: $0, 1, 1, 4, 4, 9, 9, 16, 16, 25, 25, \ldots$

**$T^2$ (unit flat torus, $[0, 2\pi)^2$):** $\lambda_{m,n} = m^2 + n^2$, mult $r_2(\lambda)$.
Ratios: $0\,(\times 1), 1\,(\times 4), 2\,(\times 4), 4\,(\times 4), 5\,(\times 8), 8\,(\times 4), 9\,(\times 4), 10\,(\times 8), 13\,(\times 8), \ldots$
Gaps at $3, 6, 7, 11, 12, 14, 15$.

**$S^2$ (unit sphere):** $\lambda_\ell = \ell(\ell+1)$, mult $2\ell+1$.
Ratios ($\div \lambda_1 = 2$): $0\,(\times 1), 1\,(\times 3), 3\,(\times 5), 6\,(\times 7), 10\,(\times 9), 15\,(\times 11), 21\,(\times 13), \ldots$
Distinctive odd-multiplicity signature.

---

## 5. Architectures

| Architecture | Role | Key hyperparameters |
|--------------|------|---------------------|
| **ReLU SAE** | Continuous baseline; predicted to succeed everywhere with enough capacity | $L_1$ coefficient $\lambda$, dictionary size $m$ |
| **TopK SAE** | Phase-transition family; primary architectural variable | $K \in \{1, 2, 3, 4, 6, 8\}$; dictionary size $m$ |
| **JumpReLU SAE** | Local thresholding without global selection — isolates whether topology destruction is from threshold or from selection | learned per-feature thresholds; $L_0$ pseudo-loss |

**Capacity matching:** for each manifold, pick $m$ large enough that ReLU consistently achieves low reconstruction error on raw samples. Use this $m$ across architectures so capacity is held constant. Use a smaller $m$ for $S^1$ to keep dead features rare.

**Negative controls:**

- **Random-init SAE** (no training): predicted to fail across the board.
- **Undersized dictionary** (e.g., $m \approx d$): predicted to fail because too few features for any reasonable representation.
- **Pre-activations on trained ReLU SAE:** predicted to recover topology, confirming the destruction mechanism is the nonlinearity.

These controls are essential: without them, you can't distinguish "method detects topology" from "method outputs topology-recovery regardless of input."

---

## 6. Predictions

Stated as falsifiable hypotheses.

### H1 — TopK phase transition at $K = d+1$

For TopK on a $d$-manifold, topological recovery (correct Betti, low spectral error) requires $K \geq d+1$. Below this threshold, post-activations are locally lower-dimensional than the manifold, forcing discontinuous active-set transitions and tearing the manifold.

**Predictions:**
- $S^1$: TopK-1 fails ($b_1 = 0$, $b_0 = K$); TopK-$K \geq 2$ succeeds.
- $T^2$, $S^2$: TopK-1, TopK-2 fail; TopK-$K \geq 3$ succeeds.
- Sweep $K$ across each manifold and plot recovery score; expect sharp transition at $K = d+1$.

### H2 — ReLU and JumpReLU succeed on all smooth manifolds (given capacity)

Any compact manifold admits a finite atlas of positive-linear charts (for ReLU), and JumpReLU's local thresholding doesn't introduce global discontinuity. Both should recover all target manifolds with sufficient $m$.

### H3 — Figure-8 reveals heterogeneous dimension

The figure-8 is 1-D away from its singular vertex but effectively 2-D at the vertex. TopK-2 should succeed on the arcs but fail at the singularity; TopK-3 should be necessary for full recovery. Within-manifold test of the paper's heterogeneous-dimension claim.

(Pre-activation recovery is *expected* but not elevated to a hypothesis — for non-degenerate $W_e$, linear maps preserve topology generically. Pre-activations remain in the negative-control set in §5 and as a diagnostic in §7.3 step 5: an SAE whose pre-activations fail to recover topology indicates a broken pipeline, not a discovery.)

---

## 7. Experimental Pipeline

### 7.1 Stage 0: Pipeline validation (do not skip)

Before any SAE is involved, run the full Mapper + Laplacian pipeline on **raw uniform samples** from each manifold. Verify:

- Mapper recovers correct $(b_0, b_1)$ with stable hyperparameter region.
- First $\sim 20$ Laplacian eigenvalue ratios match closed-form to within tolerance (target: $\mathcal{E} < 0.05$).
- Multiplicity pattern matches.

Tune $N$, $k$, $\sigma$ until validation passes. Document the working configuration as the baseline. **All SAE results downstream use this validated configuration.**

**Starting sample sizes (refine empirically):**

- $S^1$: $N = 2{,}000$
- $T^2, S^2$: $N = 10{,}000$

### 7.2 Stage 1: Train SAEs

For each (architecture, manifold) pair:

1. Generate $N_{\text{train}}$ samples (e.g., $10 \cdot N_{\text{eval}}$) lifted into ambient $\mathbb{R}^D$ with isotropic noise.
2. Train SAE to convergence. Log reconstruction loss, sparsity, dead-feature count.
3. Save encoder weights, biases, dictionary.

### 7.3 Stage 2: Extract and analyze

For each trained (architecture, manifold) pair:

1. Generate $N_{\text{eval}}$ fresh samples (different seed from training).
2. Compute $z_{\text{pre}}, z_{\text{post}}, \hat{x}$ for all samples.
3. **Mapper analysis** on $z_{\text{post}}$: three filter functions × 20-config sweep. Report $(b_0, b_1)$ and node-to-box ratio under all three filters; report nerve comparison primarily under Laplacian filter. **Headline result is from the Laplacian eigenvector filter.** GT filter result reported alongside as upper bound; PCA filter as linear baseline.
4. **Spectral analysis** on $z_{\text{post}}$: Coifman–Lafon graph Laplacian, first 20 eigenvalues, log-ratio error $\mathcal{E}$, multiplicity check, near-zero count.
5. **Diagnostic:** repeat (3) and (4) on $z_{\text{pre}}$ and $\hat{x}$ for failure cases.

### 7.4 Stage 3: Aggregate and report

Main result table: rows = architectures (incl. controls), columns = manifolds, cells = (Mapper $(b_0, b_1)$, node-to-box ratio, nerve mismatch, $\mathcal{E}$, near-zero count).

TopK-$K$ sweep figure: recovery score vs. $K$ for each manifold, looking for predicted phase transition at $K = d+1$.

Diagnostic Mapper graphs for failure cases under ground-truth filter, illustrating *how* recovery failed.

---

## 8. Further Goals

### 8.1 Disjoint unions (test global vs. local sparsity mechanisms)

| Space | Why interesting |
|-------|----------------|
| $S^1 \sqcup S^1$ | Two circles. Tests whether SAE keeps components disconnected. |
| $T^2 \sqcup T^2$ | Shared global $K$ budget — features from one torus may bleed into the other under TopK. ReLU/JumpReLU's per-feature thresholds shouldn't show this. |
| $T^2 \sqcup S^1$ | **Heterogeneous-dimension test.** TopK has to satisfy torus's $K \geq 3$ requirement while not over-activating on circle points. Direct test of the paper's heterogeneous-dimension claim, routed through geometry. |

For disjoint unions, separate components by translation in ambient space such that minimum inter-component distance $\gg$ intrinsic diameter of each component.

**Additional hypothesis (H5, future):** disjoint unions reveal global-vs-local sparsity. TopK with shared $K$ contaminates multi-component spaces; ReLU and JumpReLU don't. Spectral $b_0$ and Mapper $b_0$ should both flag this: correct = 2, contaminated = 1.

### 8.2 Higher homology

Detecting $b_2$ (e.g., the void of $S^2$) requires simplicial complexes with 2-cells; the current graph-based method only resolves $b_0, b_1$. Extension to persistent homology or Čech complexes is a separate project.

---

## 9. Open Decisions

1. **Ambient dimension $D$:** $D \in [64, 256]$. Larger = more realistic but more samples needed for kNN graphs to be meaningful. Start with $D = 64$.
2. **$\epsilon$ for multiplicity-cluster identification.** Tune during Stage 0.
3. **Whether to probe reconstructions $\hat{x}$ beyond the diagnostic case.** May reveal interesting decoder-vs-encoder asymmetries.

---

## 10. Success Criteria

The project succeeds (publishable in some form) if any of the following are demonstrated:

- **TopK phase transition at $K = d+1$** is empirically confirmed (or refuted with a clean alternative pattern).
- **Topology / geometry decoupling:** at least one architecture preserves Mapper Betti while failing spectral fidelity, illustrating the value of the two-tool approach.
- **Negative controls behave as predicted:** random-init fails everywhere, undersized-dictionary fails everywhere, pre-activations succeed where post-activations fail. (Null result for these controls would indicate the methodology is unreliable.)

The project does not require all of the above. Even a clean confirmation of H1 alone is a contribution.
