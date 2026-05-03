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
| σ used | 0.0184797 |
| Spectral K | 25 |
| Filter | Laplacian eigenvector (k=3) |

## Training (final step)

| Metric | Value |
|---|---|
| MSE | 0.003097 |
| Mean L0 | 10.11 |
| Dead atoms | 87 |

## Sign-off checklist

- [x] Mapper correct region (Laplacian filter) contains GT Betti at N=100000
- [x] Correct-region size ≥ 50% of grid (got 13/24 = 54%)
- [ ] Correct region is contiguous (n_correct_total = 14)
- [ ] Correct region is interior to grid (no edge contact)
- [ ] Failure modes outside correct region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [x] Spectral log-ratio error E < 0.05 on post-activations (E=0.0197)
- [x] Multiplicity-cluster check passes for first 4 clusters
- [x] Near-zero eigenvalue count = b₀ (got 1, expected 1)
- [x] Mapper graph renderings persisted for every config in correct region
- [x] Eigenvalues + eigenvectors + samples persisted

## Diagnostics

- Modal Betti (Laplacian filter): (1, 0)
- Expected Betti: (1, 0)

## Overall

**FAIL**
