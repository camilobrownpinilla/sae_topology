# Stage 0 Sign-off — `circle`

Generated automatically by `stage0_run.py`.

## Configuration

| Parameter | Value |
|---|---|
| N (samples) | 100000 |
| Ambient dim | 64 |
| Noise σ | 0.01 |
| kNN k (winner) | 40 |
| σ factor (winner) | 1.0 |
| σ used | 0.0947613 |
| Spectral K | 20 |

## Sign-off checklist (stage0_tuning.md §10)

- [x] Mapper stable region under Laplacian eigenvector filter contains correct Betti at N=100000
- [x] Stable region size ≥ 50% of grid (got 21/24 = 88%)
- [x] Stable region is contiguous (n_correct_anywhere = 21)
- [ ] Stable region is interior to grid (no edge contact)
- [x] Mapper failure modes outside stable region are interpretable (b₁ monotone non-decreasing in n_intervals at each fixed overlap)
- [x] Laplacian E < 0.05 on raw samples (E=0.0066)
- [x] Multiplicity-cluster check passes for first 4 clusters
- [x] Near-zero eigenvalue count = b₀ (got 1, expected 1)
- [x] Eigenvector-basis-rotation reproducibility (spectral-only) for T²/S² (n/a (S^1 — λ₁ non-degenerate))
- [x] Reference Mapper graphs persisted as renderings for every config in stable region
- [x] Reference Laplacian eigenvectors and seeds persisted
- [x] Sample point clouds and seeds persisted

## Diagnostics

- Modal Betti (Laplacian filter): (1, 1)
- GT-filter correct-region: 100% (size=24)
- PCA-filter correct-region: 100% (size=24)

## Overall

**FAIL**
