# Stage 1 Sign-off — `helix` / `relu_l1` (m=128, seed=2)

Generated automatically by `stage1_bundle.py`.

## Configuration

| Parameter | Value |
|---|---|
| Topology | helix |
| Architecture | relu_l1 |
| d_sae (m) | 128 |
| Ambient dim | 64 |
| Noise σ | 0.01 |
| Eval samples (N) | 100000 |
| Stage 0 kNN k | 25 |
| Stage 0 σ factor | 2.0 |
| σ used | 0.127673 |
| Spectral K | 25 |
| Filter | Laplacian eigenvector (k=1) |

## Training (final step)

| Metric | Value |
|---|---|
| MSE | 0.003203 |
| Mean L0 | 58.85 |
| Dead atoms | 23 |

## Sign-off checklist

- [x] Mapper correct region (Laplacian filter) contains GT Betti at N=100000
- [x] Correct-region size ≥ 50% of grid (got 22/24 = 92%)
- [x] Correct region is contiguous (n_correct_total = 22)
- [ ] Correct region is interior to grid (no edge contact)
- [x] Failure modes outside correct region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [x] Spectral log-ratio error E < 0.05 on post-activations (E=0.0126)
- [ ] Multiplicity-cluster check passes for first 4 clusters
- [x] Near-zero eigenvalue count = b₀ (got 1, expected 1)
- [x] Mapper graph renderings persisted for every config in correct region
- [x] Eigenvalues + eigenvectors + samples persisted

## Diagnostics

- Modal Betti (Laplacian filter): (1, 0)
- Expected Betti: (1, 0)

## Overall

**FAIL**
