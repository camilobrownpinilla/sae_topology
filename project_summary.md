# SAE Topological Recoverability — Project Summary

## Overview

This project investigates whether different sparse autoencoder (SAE) architectures
systematically differ in *how* they represent structured data — specifically, whether
topology encoded in the training data ends up in different parts of the SAE's learned
representation depending on architectural choices.

The central question: when an SAE is trained on data that lies on a manifold (a circle,
two circles, a figure-eight), does the manifold geometry get embedded in the decoder
atoms, in the sparse codes, or neither?

---

## Core Conceptual Framework

A trained SAE produces two objects for each input $x$:

- **Decoder atoms** $\{d_i\} \subset \mathbb{R}^d$: the dictionary of learned features,
  fixed after training.
- **Sparse codes** $z(x) \in \mathbb{R}^m$: the per-sample activation vector, with most
  entries zero.

The project defines two possible locations for topological structure:

- **T1 (atom-located):** The decoder atoms, viewed as a point cloud in $\mathbb{R}^d$,
  inherit the topology of the training manifold. A SAE trained on circle-distributed data
  would have atoms arranged in a ring.
- **T2 (code-located):** The sparse codes, viewed as a point cloud in $\mathbb{R}^m$
  (after PCA reduction), trace out the training manifold. As inputs move around the
  circle, the pattern of which atoms fire traces a loop.

These are measured using **persistent homology (PH)** — a method from topological data
analysis that detects loops ($H_1$ cycles) in point clouds robustly under noise. The
**T1/T2 score** is defined as $\rho_{T1} / (\rho_{T1} + \rho_{T2})$, where each $\rho$
is the normalized $H_1$ persistence of the respective point cloud. A score near 1
indicates T1 dominance; near 0 indicates T2 dominance.

---

## Architectural Bias Hypothesis

The project's core hypothesis is that the sparsity mechanism shapes which representation
form is preferred:

- **ReLU + L1:** The L1 penalty discourages coactivation. Nearby inputs on the manifold
  cannot jointly activate many atoms, so the SAE is pushed to tile the manifold with
  individual atoms (T1). The geometry lives in the *dictionary*, not in the *pattern of
  activations*.
- **TopK:** By allowing exactly $k$ atoms to coactivate per sample with no penalty,
  the SAE can represent manifold position through which *combination* of atoms fires.
  This opens the door to T2: as inputs move around the manifold, the activation pattern
  sweeps out a topological structure in code space.
- **JumpReLU:** Expected to be intermediate, with a learnable threshold that modulates
  between these regimes. (Scaffolded; not yet fully evaluated.)

---

## Methodology

### Data Generating Processes

Synthetic data is sampled from known topological spaces embedded in $\mathbb{R}^d$
($d = 8$) with Gaussian noise ($\sigma = 0.05$). Topologies studied:

- **Circle** ($S^1$): unit circle randomly oriented in $\mathbb{R}^8$; expected $H_1 = 1$.
- **Two Circles**: two disjoint unit circles with separation 5.0; expected $H_1 = 2$,
  $H_0 = 2$.
- **Figure Eight**: two circles sharing a wedge point; expected $H_1 = 2$, $H_0 = 1$.
- **Torus** ($T^2$): excluded from cross-topology sweep pending validation.

### Persistent Homology

PH is computed via Vietoris-Rips filtration using `ripser`, up to $H_1$. A critical
calibration finding: a noise floor of $\text{min\_persistence} = 3\sigma\sqrt{d} \approx
0.424$ must be applied when filtering bars, otherwise Gaussian noise in high-dimensional
PCA projections produces spurious $H_1$ components that contaminate T2 scores.

### SAE Training

SAEs are trained on online DGP samples (fresh batches each step, no dataset) for 30,000
steps. Decoder atoms are constrained to unit norm throughout training. Dead-atom
resampling (via EMA activation tracking) recovers dormant atoms by reinitializing them
from fresh DGP samples.

A key calibration result: for circle topology with $m = 64$ atoms, the L1 coefficient
must be $\lambda = 0.02$ to achieve the T1-dominant regime. At $\lambda = 0.001$ (10×
weaker), the penalty is too soft — activations remain dense ($L_0 \approx 20$), and even
ReLU+L1 SAEs show T2-dominant codes. Matching sparsity: both architectures were compared
at $L_0 \approx 9$ (relu\_l1 at $\lambda = 0.02$; topk at $k = 9$).

---

## Results

### Stage 2: Hypothesis Validation on Circle

Training relu\_l1 and topk on the circle topology at the calibrated settings:

| Architecture | $L_0$ | T1/T2 Score | Interpretation |
|---|---|---|---|
| relu\_l1 ($\lambda = 0.02$) | 8.6 | **1.000** | Pure T1: atoms form a ring, codes are unstructured |
| topk ($k = 9$) | 9.0 | **0.211** | T2-permissive: codes carry the loop |

The hypothesis is strongly confirmed on this topology. The rho\_T2 for relu\_l1 was `nan`
(no H1 signal in codes whatsoever); topk showed clear H1 in code space.

### Stage 3: Cross-Topology Sweep

Running both architectures across three topologies ($m = 64$, 3 seeds each):

| Topology | relu\_l1 score | topk score | Result |
|---|---|---|---|
| Circle | 1.000 ± 0.000 | 0.201 ± 0.079 | Hypothesis confirmed |
| Two Circles | 0.585 ± 0.106 | **0.823 ± 0.093** | Hypothesis **reversed** |
| Figure Eight | 0.031 ± 0.010 | 0.000 ± 0.000 | Both fail |

---

## Interpretation of Negative Results

### Two Circles: Reversal

The hypothesis reversal (topk outperforming relu\_l1 on T1) is attributed to the
interaction between the L1 coefficient calibration and the geometry of the two-circle
DGP.

The two circles are separated by 5.0 units, meaning the data spans a region roughly 5×
larger than the unit circle used for calibration. At this scale, the L1 penalty
($\lambda = 0.02$) is disproportionately strong relative to the manifold signal: it
aggressively suppresses activations in a regime where the coordinate magnitudes are much
larger, causing atoms to cluster rather than spread across each ring.

TopK, lacking any global sparsity penalty, naturally allocates its fixed budget of $k = 9$
atoms per sample within whichever cluster the sample belongs to. The geometric separation
already prevents cross-cluster coactivation regardless of architecture. Within each
cluster, TopK's forced diversity of the active set causes atoms to spread around the ring
— producing clean T1 structure.

**Key implication:** the L1 coefficient is not a universal hyperparameter. It must be
recalibrated to the scale of the training manifold, not just to the desired sparsity level.

### Figure Eight: Universal Failure

Both architectures achieve near-zero T1/T2 scores on the figure-eight topology. The most
likely cause is the wedge point — the shared junction where both loops meet. Atoms near
the wedge must serve both loops simultaneously, preventing the formation of two clean
rings in atom space. The same ambiguity affects T2: codes for near-wedge inputs cannot
cleanly be assigned to one loop or the other, disrupting the code-space topology.

The figure-eight result may also reflect a known methodological risk: finite-sample PH
near a wedge point is fragile, as points cluster densely at the junction and can
short-circuit $H_1$ bars.

---

## Current Status and Open Questions

**Confirmed:**
- The T1/T2 scoring framework is valid (sanity checks pass; circle results are clean and
  reproducible across seeds).
- The architectural bias hypothesis holds robustly for the circle topology.
- L1 calibration is critical: the "correct" $\lambda$ depends on both the desired $L_0$
  and the geometric scale of the training data.

**Open questions:**
1. Does the two-circles reversal disappear when $\lambda$ is rescaled to the manifold
   diameter? (Proposed: use $\lambda = 0.002$ for separation = 5.0.)
2. Is the figure-eight failure due to geometry (wedge point) or PH fragility? Inspecting
   decoder atom PCA visually would distinguish these.
3. How does the hypothesis generalize to higher-genus topologies (torus, $H_1 = 2$,
   $H_2 = 1$)?
4. Does JumpReLU behave closer to TopK or to ReLU+L1 as the learnable threshold adapts?

**Planned next stages:**
- Per-topology L1 recalibration to control for scale effects.
- Visual inspection of two-circles and figure-eight decoder atoms.
- Full architecture sweep (relu\_l1, topk, jumprelu) once calibration confounds are resolved.
- Ablations over noise $\sigma$, ambient dimension $d$, and sparsity level $k / \lambda$.
