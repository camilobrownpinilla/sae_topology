"""Three Mapper filter functions, per spec section 4.1.

  - laplacian_eigenvector_filter (primary): first k non-trivial eigenvectors
    of the Coifman-Lafon graph Laplacian on Z. Derives the cover from the
    data, so genuinely tests whether the SAE preserves enough geometric
    structure for a data-driven coordinate system to rediscover the manifold.

  - ground_truth_filter (diagnostic / upper bound): intrinsic coordinates of
    the source manifold. Available only because of the toy-model setup.

  - pca_filter (linear baseline): top-k principal components of Z.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from ..spectral.graph_laplacian import coifman_lafon_spectrum


def laplacian_eigenvector_filter(
    Z: np.ndarray,
    k: int,
    knn_k: int = 15,
    eigenvectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the first k non-trivial Laplacian eigenvectors of Z, shape (N, k).

    If `eigenvectors` is supplied (the (N, K) eigenvector matrix from a prior
    `coifman_lafon_spectrum` call with K >= k+1), the trivial column 0 is
    dropped and the next k are returned. Otherwise the spectrum is computed
    fresh.
    """
    if eigenvectors is None:
        spec = coifman_lafon_spectrum(Z, knn_k=knn_k, K=k + 1)
        eigvecs = spec['eigenvectors']
    else:
        eigvecs = np.asarray(eigenvectors)
        if eigvecs.shape[1] < k + 1:
            raise ValueError(
                f"Provided eigenvectors have {eigvecs.shape[1]} columns; "
                f"need at least k+1={k+1} (column 0 is the trivial eigenvector)."
            )
    return eigvecs[:, 1:k + 1]


def ground_truth_filter(gt_coords: Optional[np.ndarray]) -> np.ndarray:
    """Return the intrinsic coordinates of the source manifold as the filter.

    Raises ValueError when gt_coords is None (manifold has no surfaced
    intrinsic coordinate system - e.g., FigureEight).
    """
    if gt_coords is None:
        raise ValueError(
            "ground_truth_filter requires gt_coords; "
            "manifold has no surfaced intrinsic coordinate system."
        )
    gt = np.asarray(gt_coords, dtype=float)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    return gt


def pca_filter(Z: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k principal components of Z, shape (N, k)."""
    return PCA(n_components=k).fit_transform(Z)
