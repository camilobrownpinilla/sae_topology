from typing import Optional, Tuple

import numpy as np

from .base import TopologicalSpace


def _gram_schmidt_pair(d: int):
    """Return two orthonormal vectors in R^d chosen uniformly at random."""
    u = np.random.randn(d)
    u /= np.linalg.norm(u)
    v = np.random.randn(d)
    v -= v.dot(u) * u
    v /= np.linalg.norm(v)
    return u, v


def _gram_schmidt_k(d: int, k: int) -> list:
    """Return k mutually orthonormal random vectors in R^d."""
    basis = []
    for _ in range(k):
        v = np.random.randn(d)
        for b in basis:
            v -= v.dot(b) * b
        v /= np.linalg.norm(v)
        basis.append(v)
    return basis


class Circle(TopologicalSpace):
    """Unit circle randomly oriented in R^d.  Expected: H_0=1, H_1=1.

    `sample_with_gt` returns the canonical 2D embedding (cos theta, sin theta)
    as gt_coords - shape (n, 2). This is the natural Mapper ground-truth
    filter: Mapper covers the filter image with axis-aligned boxes, and
    the image of (cos theta, sin theta) is S^1 in R^2, which Mapper
    correctly resolves into a cycle. (A 1D theta filter on [0, 2pi)
    would NOT detect the cycle, since the cover is non-periodic.)
    Intrinsic theta can be recovered as np.arctan2(gt[:, 1], gt[:, 0]).
    """

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray):
        super().__init__(dimension, noise_level)
        self.center = np.asarray(center, dtype=float)
        self.u, self.v = _gram_schmidt_pair(dimension)

    def sample(self, n: int) -> np.ndarray:
        return self.sample_with_gt(n)[0]

    def sample_with_gt(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.random.uniform(0, 2 * np.pi, n)
        coords = np.outer(np.cos(thetas), self.u) + np.outer(np.sin(thetas), self.v)
        coords = coords + self.center + np.random.randn(n, self.dimension) * self.noise_level
        gt = np.column_stack([np.cos(thetas), np.sin(thetas)])
        return coords, gt


class Points(TopologicalSpace):
    """k isolated Gaussian clusters in R^d.  Expected: H_0=k, H_1=0."""

    def __init__(self, dimension: int, noise_level: float,
                 n_points: int = 3, spread: float = 5.0):
        super().__init__(dimension, noise_level)
        self.n_points = n_points
        dirs = [np.random.randn(dimension) for _ in range(n_points)]
        self.centers = np.array([d / np.linalg.norm(d) * spread for d in dirs])

    def sample(self, n: int) -> np.ndarray:
        k = self.n_points
        parts = []
        for i, c in enumerate(self.centers):
            n_i = n // k if i < k - 1 else n - (n // k) * (k - 1)
            parts.append(c + np.random.randn(n_i, self.dimension) * self.noise_level)
        return np.vstack(parts)


class TwoCircles(TopologicalSpace):
    """Two disjoint unit circles at a given separation.  Expected: H_0=2, H_1=2."""

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray,
                 separation: float = 5.0, radii: tuple = (1.0, 1.0)):
        super().__init__(dimension, noise_level)
        self.radii = radii
        self.c1 = np.asarray(center, dtype=float)
        direction = np.random.randn(dimension)
        direction /= np.linalg.norm(direction)
        self.c2 = self.c1 + separation * direction
        self.u1, self.v1 = _gram_schmidt_pair(dimension)
        self.u2, self.v2 = _gram_schmidt_pair(dimension)

    def sample(self, n: int) -> np.ndarray:
        n1, n2 = n // 2, n - n // 2
        r1, r2 = self.radii

        t1 = np.random.uniform(0, 2 * np.pi, n1)
        pts1 = (r1 * np.outer(np.cos(t1), self.u1)
                + r1 * np.outer(np.sin(t1), self.v1)
                + self.c1
                + np.random.randn(n1, self.dimension) * self.noise_level)

        t2 = np.random.uniform(0, 2 * np.pi, n2)
        pts2 = (r2 * np.outer(np.cos(t2), self.u2)
                + r2 * np.outer(np.sin(t2), self.v2)
                + self.c2
                + np.random.randn(n2, self.dimension) * self.noise_level)

        return np.vstack([pts1, pts2])


class FigureEight(TopologicalSpace):
    """Two circles sharing a wedge point.  Expected: H_0=1, H_1=2.

    Each lobe has center = wedge + radius * u_i. At theta=pi both lobes
    pass through `center` (the wedge), so they are joined. u1 and u2 are
    orthogonal so the lobes extend in different directions from the wedge.
    """

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray,
                 radius: float = 1.0):
        super().__init__(dimension, noise_level)
        self.wedge = np.asarray(center, dtype=float)
        self.radius = radius

        u1 = np.random.randn(dimension)
        u1 /= np.linalg.norm(u1)

        u2 = np.random.randn(dimension)
        u2 -= u2.dot(u1) * u1
        u2 /= np.linalg.norm(u2)

        v1 = np.random.randn(dimension)
        v1 -= v1.dot(u1) * u1
        v1 /= np.linalg.norm(v1)

        v2 = np.random.randn(dimension)
        v2 -= v2.dot(u2) * u2
        v2 -= v2.dot(u1) * u1
        norm2 = np.linalg.norm(v2)
        if norm2 < 1e-8:
            v2 = np.random.randn(dimension)
            v2 -= v2.dot(u2) * u2
            v2 /= np.linalg.norm(v2)
        else:
            v2 /= norm2

        self.u1, self.v1 = u1, v1
        self.u2, self.v2 = u2, v2
        self.c1 = self.wedge + radius * u1
        self.c2 = self.wedge + radius * u2

    def sample(self, n: int) -> np.ndarray:
        n1, n2 = n // 2, n - n // 2

        t1 = np.random.uniform(0, 2 * np.pi, n1)
        pts1 = (self.radius * np.outer(np.cos(t1), self.u1)
                + self.radius * np.outer(np.sin(t1), self.v1)
                + self.c1
                + np.random.randn(n1, self.dimension) * self.noise_level)

        t2 = np.random.uniform(0, 2 * np.pi, n2)
        pts2 = (self.radius * np.outer(np.cos(t2), self.u2)
                + self.radius * np.outer(np.sin(t2), self.v2)
                + self.c2
                + np.random.randn(n2, self.dimension) * self.noise_level)

        return np.vstack([pts1, pts2])


class Torus(TopologicalSpace):
    """Flat-torus embedding in R^d.  Expected: H_0=1, H_1=2, H_2=1.

    Parametric map: (phi, theta) ->
        R*(cos phi)*u1 + R*(sin phi)*u2 + r*(cos theta)*v1 + r*(sin theta)*v2
    where u1, u2, v1, v2 are 4 mutually orthonormal vectors in R^d.

    For the spec's unit flat torus (LB eigenvalues lambda_{m,n} = m^2 + n^2)
    use major_radius = minor_radius = 1.0. The notebook-era default
    (R=2, r=1) is preserved here so the existing notebook keeps working;
    the new pipeline configs override to R=r=1.
    """

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray,
                 major_radius: float = 2.0, minor_radius: float = 1.0):
        super().__init__(dimension, noise_level)
        self.center = np.asarray(center, dtype=float)
        self.R = major_radius
        self.r = minor_radius
        self.u1, self.u2, self.v1, self.v2 = _gram_schmidt_k(dimension, 4)

    def sample(self, n: int) -> np.ndarray:
        return self.sample_with_gt(n)[0]

    def sample_with_gt(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        phi   = np.random.uniform(0, 2 * np.pi, n)   # longitude
        theta = np.random.uniform(0, 2 * np.pi, n)   # meridian
        pts = (self.R * np.outer(np.cos(phi),   self.u1) +
               self.R * np.outer(np.sin(phi),   self.u2) +
               self.r * np.outer(np.cos(theta), self.v1) +
               self.r * np.outer(np.sin(theta), self.v2))
        coords = self.center + pts + np.random.randn(n, self.dimension) * self.noise_level
        # Canonical 4D Clifford embedding as gt_coords; Mapper covers
        # this in R^4, which correctly resolves the toroidal grid.
        # A raw (theta, phi) in [0, 2pi)^2 filter would NOT, because the
        # cover is non-periodic.
        gt = np.column_stack([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)])
        return coords, gt


class Sphere(TopologicalSpace):
    """Unit S^2 randomly oriented in R^d.  Expected: H_0=1, H_1=0, H_2=1.

    Sample x ~ N(0, I_3) then normalize to unit length (uniform on S^2 by
    rotational invariance), embed into R^d via 3 mutually orthonormal
    vectors u1, u2, u3:

        embedded(x) = x[0]*u1 + x[1]*u2 + x[2]*u3

    The intrinsic coordinates returned by `sample_with_gt` are the 3D
    Cartesian coords on the unit sphere itself (per spec section 3.1
    line 70: "the 3D Cartesian (x,y,z) directly - injective on S^2,
    avoiding the polar singularities of spherical coordinates"). These
    are degree-1 spherical harmonics, equal to the first three non-trivial
    Laplace-Beltrami eigenfunctions on S^2.
    """

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray,
                 radius: float = 1.0):
        super().__init__(dimension, noise_level)
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.u1, self.u2, self.u3 = _gram_schmidt_k(dimension, 3)

    def sample(self, n: int) -> np.ndarray:
        return self.sample_with_gt(n)[0]

    def sample_with_gt(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        x3 = np.random.randn(n, 3)
        x3 /= np.linalg.norm(x3, axis=1, keepdims=True)
        x3 *= self.radius
        pts = (np.outer(x3[:, 0], self.u1)
               + np.outer(x3[:, 1], self.u2)
               + np.outer(x3[:, 2], self.u3))
        coords = self.center + pts + np.random.randn(n, self.dimension) * self.noise_level
        return coords, x3


class Helix(TopologicalSpace):
    """Open helix (1D, contractible) randomly oriented in R^d.  Expected: H_0=1, H_1=0.

    Parametric map: t in [0, 2*pi*n_turns] ->
        radius * cos(t) * u1 + radius * sin(t) * u2 + (pitch * t) * u3
    where u1, u2, u3 are 3 mutually orthonormal vectors in R^d.

    Constant-speed in t (ds/dt = sqrt(R^2 + c^2)), so uniform t is uniform
    in arc length. Total arc length L = 2*pi*n_turns*sqrt(R^2 + c^2).

    Riemannian-isometric to a line segment of length L; the LB spectrum
    (Neumann BCs) is therefore lambda_n = (n*pi/L)^2, all multiplicity 1.
    The normalized ratios are [0, 1, 4, 9, ...] = [n^2 for n in 0..]:
    numerically the same as S^1 levels but with multiplicity 1 instead
    of 2, which `multiplicity_check` distinguishes.

    `sample_with_gt` returns the raw arc-length parameter s = sqrt(R^2 + c^2)*t
    as gt_coords - shape (n, 1). Mapper covers the 1-D image with
    axis-aligned intervals, which correctly resolves a contractible line.
    """

    def __init__(self, dimension: int, noise_level: float, center: np.ndarray,
                 radius: float = 1.0, pitch: float = 0.5, n_turns: float = 4.0):
        super().__init__(dimension, noise_level)
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.pitch = pitch
        self.n_turns = n_turns
        self.t_max = 2 * np.pi * n_turns
        self.u1, self.u2, self.u3 = _gram_schmidt_k(dimension, 3)

    def sample(self, n: int) -> np.ndarray:
        return self.sample_with_gt(n)[0]

    def sample_with_gt(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.random.uniform(0.0, self.t_max, n)
        pts = (self.radius * np.outer(np.cos(t), self.u1)
               + self.radius * np.outer(np.sin(t), self.u2)
               + np.outer(self.pitch * t, self.u3))
        coords = self.center + pts + np.random.randn(n, self.dimension) * self.noise_level
        s = np.sqrt(self.radius * self.radius + self.pitch * self.pitch) * t
        return coords, s.reshape(-1, 1)


def make_dgp(topology: str, d: int, sigma: float, c0: np.ndarray,
             N: int = 0, **kwargs) -> TopologicalSpace:
    """Factory: return the TopologicalSpace for the requested topology.

    Args:
        topology:  'points' | 'circle' | 'two_circles' | 'figure_eight' | 'torus' | 'sphere' | 'helix'
        d:         ambient dimension
        sigma:     noise level
        c0:        center array of shape (d,)
        N:         unused (kept for call-site consistency; call dgp.sample(N))
        **kwargs:
            points       — n_points (int=3), spread (float=5.0)
            two_circles  — separation (float=5.0), radii (tuple=(1.,1.))
            figure_eight — radius (float=1.0)
            torus        — major_radius (float=2.0), minor_radius (float=1.0)
            sphere       — radius (float=1.0)
            helix        — radius (float=1.0), pitch (float=0.5), n_turns (float=4.0)
    """
    c0 = np.asarray(c0, dtype=float)
    if topology == 'points':
        return Points(d, sigma,
                      n_points=kwargs.get('n_points', 3),
                      spread=kwargs.get('spread', 5.0))
    if topology == 'circle':
        return Circle(d, sigma, c0)
    if topology == 'two_circles':
        return TwoCircles(d, sigma, c0,
                          separation=kwargs.get('separation', 5.0),
                          radii=kwargs.get('radii', (1.0, 1.0)))
    if topology == 'figure_eight':
        return FigureEight(d, sigma, c0, radius=kwargs.get('radius', 1.0))
    if topology == 'torus':
        return Torus(d, sigma, c0,
                     major_radius=kwargs.get('major_radius', 2.0),
                     minor_radius=kwargs.get('minor_radius', 1.0))
    if topology == 'sphere':
        return Sphere(d, sigma, c0, radius=kwargs.get('radius', 1.0))
    if topology == 'helix':
        return Helix(d, sigma, c0,
                     radius=kwargs.get('radius', 1.0),
                     pitch=kwargs.get('pitch', 0.5),
                     n_turns=kwargs.get('n_turns', 4.0))
    raise ValueError(
        f"Unknown topology '{topology}'. "
        "Choose from: 'points', 'circle', 'two_circles', 'figure_eight', 'torus', 'sphere', 'helix'."
    )
