"""Coifman-Lafon alpha=1 graph Laplacian on a kNN graph with Gaussian
edge weights. Per spec section 4.3.

The pipeline:
  1. kNN graph on Z (Euclidean distances).
  2. Gaussian weights W_ij = exp(-||z_i - z_j||^2 / (2 sigma^2)) on each edge,
     with sigma = median pairwise distance among kNN edges.
  3. Coifman-Lafon alpha=1 normalisation:  W_tilde = D^-1 W D^-1.
  4. Symmetric normalised Laplacian:  L_sym = I - D_tilde^{-1/2} W_tilde D_tilde^{-1/2}.
  5. First K eigenvalues via scipy.sparse.linalg.eigsh.

Returns the symmetric form (eigenvalues identical to the random-walk form
L_rw = I - D_tilde^{-1} W_tilde given in the spec); using L_sym keeps eigsh
numerically well-behaved.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors


# Hard cap above which the dense N^3 fallback is unsafe (~80 GB at N=100k).
DENSE_FALLBACK_MAX_N = 20_000


def compute_knn(Z: np.ndarray, knn_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kNN distances+indices with the self-neighbor stripped.

    Returns (distances, indices) each shape (N, knn_k). Useful as a
    standalone helper so the same kNN can be reused across different
    sigma_factor / spectrum calls (sigma only changes Gaussian weights,
    not the underlying graph).
    """
    Z = np.asarray(Z, dtype=float)
    N = len(Z)
    if knn_k >= N:
        raise ValueError(f"knn_k ({knn_k}) must be < N ({N}).")
    nn = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='auto')
    nn.fit(Z)
    distances, indices = nn.kneighbors(Z)
    return distances[:, 1:], indices[:, 1:]


def build_coifman_lafon(
    Z: np.ndarray,
    knn_k: int = 25,
    sigma: Optional[float] = None,
    sigma_factor: float = 1.0,
    precomputed_knn: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Build the Coifman-Lafon graph Laplacian on Z.

    Args:
        Z:               (N, m) array of points.
        knn_k:           number of nearest neighbors per node (excluding self).
        sigma:           Gaussian kernel bandwidth. If None, computed as
                         `sigma_factor * median(kNN edge distances)`.
        sigma_factor:    scale factor on the median-kNN heuristic. Has no
                         effect when `sigma` is supplied explicitly.
        precomputed_knn: optional (distances, indices) tuple as returned by
                         `compute_knn(Z, knn_k)`. When supplied, the kNN
                         computation is skipped. The graph topology depends
                         only on (Z, knn_k); sigma_factor variants of the
                         same (Z, knn_k) can share the kNN to save the
                         O(N log N) NearestNeighbors call.

    Returns dict:
        L_sym:        scipy.sparse symmetric normalised Laplacian.
        W:            scipy.sparse Gaussian-weighted symmetric kNN graph.
        sigma_used:   float bandwidth that was used.
        knn_k:        number of nearest neighbors used.
        N:            number of nodes.
    """
    Z = np.asarray(Z, dtype=float)
    N = len(Z)
    if knn_k >= N:
        raise ValueError(f"knn_k ({knn_k}) must be < N ({N}).")

    if precomputed_knn is not None:
        distances, indices = precomputed_knn
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if distances.shape != (N, knn_k) or indices.shape != (N, knn_k):
            raise ValueError(
                f"precomputed_knn shape mismatch: expected ({N}, {knn_k}), "
                f"got distances {distances.shape}, indices {indices.shape}."
            )
    else:
        distances, indices = compute_knn(Z, knn_k)

    if sigma is None:
        sigma = sigma_factor * float(np.median(distances))
        if sigma <= 0:
            raise ValueError("Median kNN distance is zero; check Z for duplicates.")

    weights = np.exp(-(distances ** 2) / (2.0 * sigma * sigma))

    rows = np.repeat(np.arange(N), knn_k)
    cols = indices.flatten()
    data = weights.flatten()
    W = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)

    deg = np.asarray(W.sum(axis=1)).flatten()
    deg_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
    W_tilde = deg_inv @ W @ deg_inv

    deg_tilde = np.asarray(W_tilde.sum(axis=1)).flatten()
    deg_tilde_isqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg_tilde, 1e-12)))
    L_sym = sp.identity(N) - deg_tilde_isqrt @ W_tilde @ deg_tilde_isqrt
    L_sym = (L_sym + L_sym.T) * 0.5

    return {
        'L_sym': L_sym.tocsr(),
        'W': W,
        'sigma_used': sigma,
        'knn_k': knn_k,
        'N': N,
    }


def _deterministic_v0(N: int) -> np.ndarray:
    """Deterministic starting vector for ARPACK so that eigsh is reproducible
    across processes (loky workers don't share numpy RNG state).

    Must NOT be the constant vector: every graph Laplacian satisfies L · 1 = 0,
    so a constant v0 lies exactly in the kernel and ARPACK aborts with error
    -9 ('Starting vector is zero') after the first orthogonalisation. We use
    a fixed-seed RNG draw — generic enough to have nonzero projection onto
    every eigenvector with probability 1, while still bit-identical across
    runs and processes.
    """
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(N)
    v0 /= np.linalg.norm(v0)
    return v0


def laplacian_eigendecomposition(
    L: sp.spmatrix,
    K: int = 20,
    which: str = 'SA',
) -> dict:
    """Compute the K smallest eigenvalues / eigenvectors of L.

    Robustness cascade (sparse-only at large N):

      1. eigsh(L, k=K, which='SA', v0=ones/sqrt(N))               -- primary
      2. on ArpackNoConvergence: eigsh shift-invert at sigma=0    -- robust on
         near-singular graph Laplacians (TopK collapse, etc.)
      3. on second failure: eigsh SA with maxiter=5000, tol=1e-9
      4. last resort:
           - if N < DENSE_FALLBACK_MAX_N (20k): dense eigh + warn
           - else: raise RuntimeError (dense at N>=20k is OOM-prone:
             N=100k dense Laplacian = ~80 GB)

    `which='SA'` (smallest algebraic) is more robust than 'SM' for graph
    Laplacians (which have a 0 eigenvalue that confuses ARPACK's
    smallest-magnitude search). The explicit `v0` makes eigsh deterministic
    across processes (otherwise ARPACK uses the local numpy RNG, which
    differs across loky workers even with the same seed). v0 is drawn from
    a fixed-seed RNG rather than the constant vector, since the constant
    vector is in the kernel of every graph Laplacian and trips ARPACK
    error -9 ('Starting vector is zero').
    """
    N = L.shape[0]
    K = min(K, N - 1)
    v0 = _deterministic_v0(N)

    try:
        eigvals, eigvecs = spla.eigsh(
            L, k=K, which=which, v0=v0, maxiter=50000, tol=1e-9,
        )
    except spla.ArpackNoConvergence:
        try:
            eigvals, eigvecs = spla.eigsh(
                L, k=K, sigma=0.0, which='LM', mode='normal', v0=v0,
            )
        except (spla.ArpackNoConvergence, RuntimeError):
            try:
                eigvals, eigvecs = spla.eigsh(
                    L, k=K, which=which, v0=v0, maxiter=5000, tol=1e-9,
                )
            except spla.ArpackNoConvergence:
                if N < DENSE_FALLBACK_MAX_N:
                    warnings.warn(
                        f"laplacian_eigendecomposition: ARPACK failed at N={N}; "
                        f"falling back to dense eigh ({(N**2)*8/1e9:.1f} GB allocation).",
                        RuntimeWarning,
                    )
                    L_dense = L.toarray()
                    all_vals, all_vecs = np.linalg.eigh(L_dense)
                    eigvals = all_vals[:K]
                    eigvecs = all_vecs[:, :K]
                else:
                    raise RuntimeError(
                        f"laplacian_eigendecomposition: ARPACK failed at N={N}; "
                        f"dense fallback is disabled for N>={DENSE_FALLBACK_MAX_N} "
                        f"(would allocate ~{(N**2)*8/1e9:.0f} GB). Investigate L "
                        f"degeneracy (rank deficit, disconnected components, "
                        f"collapsed-feature SAE)."
                    )

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals[eigvals < 0] = 0.0
    return {'eigenvalues': eigvals, 'eigenvectors': eigvecs}


def coifman_lafon_spectrum(
    Z: np.ndarray,
    knn_k: int = 25,
    K: int = 20,
    sigma: Optional[float] = None,
    sigma_factor: float = 1.0,
    precomputed_knn: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Convenience: build CL Laplacian on Z and return its first K eigenvalues
    + eigenvectors plus the underlying graph (so a Mapper Laplacian-eigenvector
    filter can reuse the same computation).

    `precomputed_knn` is forwarded to `build_coifman_lafon`; supply it to
    share kNN structure across multiple sigma_factor variants.
    """
    cl = build_coifman_lafon(
        Z, knn_k=knn_k, sigma=sigma, sigma_factor=sigma_factor,
        precomputed_knn=precomputed_knn,
    )
    eig = laplacian_eigendecomposition(cl['L_sym'], K=K)
    return {
        'eigenvalues': eig['eigenvalues'],
        'eigenvectors': eig['eigenvectors'],
        'L_sym': cl['L_sym'],
        'W': cl['W'],
        'sigma_used': cl['sigma_used'],
        'knn_k': cl['knn_k'],
        'N': cl['N'],
    }
