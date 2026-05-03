# Stage 1 Sign-off — `sphere` / `relu_l1` (m=128, seed=2)

Generated automatically by `stage1_bundle.py`.

## Configuration

| Parameter | Value |
|---|---|
| Topology | sphere |
| Architecture | relu_l1 |
| d_sae (m) | 128 |
| Ambient dim | 64 |
| Noise σ | 0.01 |
| Eval samples (N) | 100000 |
| Stage 0 kNN k | 40 |
| Stage 0 σ factor | 1.0 |
| σ used | 0.0279256 |
| Spectral K | 25 |
| Filter | Laplacian eigenvector (k=3) |

## Training (final step)

| Metric | Value |
|---|---|
| MSE | 0.0002987 |
| Mean L0 | 19.84 |
| Dead atoms | 44 |

## Sign-off checklist

- [x] Mapper correct region (Laplacian filter) contains GT Betti at N=100000
- [x] Correct-region size ≥ 50% of grid (got 15/24 = 62%)
- [x] Correct region is contiguous (n_correct_total = 15)
- [ ] Correct region is interior to grid (no edge contact)
- [x] Failure modes outside correct region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [ ] Spectral log-ratio error E < 0.05 on post-activations (E=0.0502)
- [ ] Multiplicity-cluster check passes for first 4 clusters
- [x] Near-zero eigenvalue count = b₀ (got 1, expected 1)
- [x] Mapper graph renderings persisted for every config in correct region
- [x] Eigenvalues + eigenvectors + samples persisted

## Diagnostics

- Modal Betti (Laplacian filter): (1, 0)
- Expected Betti: (1, 0)

## Overall

**FAIL**
