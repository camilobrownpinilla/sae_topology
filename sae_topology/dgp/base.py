from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class TopologicalSpace(ABC):
    """Abstract base class for synthetic activation data generators.

    Each subclass represents one topological shape. Callers instantiate the
    shape once (which fixes the random orientation in R^d) then call
    `sample(n)` repeatedly to draw independent noisy point clouds.

    Subclasses with a meaningful intrinsic coordinate system (e.g. theta on
    S^1, (theta, phi) on T^2, Cartesian (x,y,z) on S^2) override
    `sample_with_gt(n)` to expose those coordinates alongside the noisy
    ambient samples. The default implementation falls back to `sample(n)`
    with `gt_coords=None` for spaces (FigureEight, Points, TwoCircles)
    where intrinsic coordinates are not meaningful or are left unsurfaced.
    """

    def __init__(self, dimension: int, noise_level: float):
        self.dimension = dimension
        self.noise_level = noise_level

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Return an (n, dimension) array of noisy points from this space."""

    def sample_with_gt(self, n: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (samples, gt_coords) where gt_coords are intrinsic coordinates
        on M, or None for spaces without a surfaced intrinsic coordinate system.

        Default falls back to `sample(n)` with no gt_coords. Override in
        subclasses that have closed-form intrinsic coordinates the Mapper
        ground-truth filter can use.
        """
        return self.sample(n), None
