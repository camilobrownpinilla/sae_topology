# Stage 1 Sign-off — `torus` / `relu_l1` (m=128, seed=0)

Generated automatically by `stage1_bundle.py`.

## Configuration

| Parameter | Value |
|---|---|
| Topology | torus |
| Architecture | relu_l1 |
| d_sae (m) | 128 |
| Ambient dim | 64 |
| Noise σ | 0.01 |
| Eval samples (N) | 100000 |
| Stage 0 kNN k | 40 |
| Stage 0 σ factor | 0.5 |
| σ used | 0.0146574 |
| Spectral K | 25 |
| Filter | Laplacian eigenvector (k=4) |

## Training (final step)

| Metric | Value |
|---|---|
| MSE | 0.003337 |
| Mean L0 | 17.67 |
| Dead atoms | 35 |

## Sign-off checklist

- [ ] Mapper correct region (Laplacian filter) contains GT Betti at N=100000
- [ ] Correct-region size ≥ 50% of grid (got 1/24 = 4%)
- [ ] Correct region is contiguous (n_correct_total = 2)
- [ ] Correct region is interior to grid (no edge contact)
- [x] Failure modes outside correct region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [x] Spectral log-ratio error E < 0.05 on post-activations (E=0.0383)
- [ ] Multiplicity-cluster check passes for first 4 clusters
- [x] Near-zero eigenvalue count = b₀ (got 1, expected 1)
- [x] Mapper graph renderings persisted for every config in correct region
- [x] Eigenvalues + eigenvectors + samples persisted

## Diagnostics

- Modal Betti (Laplacian filter): (1, 2)
- Expected Betti: (1, 2)

## Overall

**FAIL**
