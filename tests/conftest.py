"""Shared fixtures: deterministic small-N samples per topology."""
from __future__ import annotations

import numpy as np
import pytest

from sae_topology.dgp import make_dgp


def _sample(topology: str, n: int, d: int, sigma: float, seed: int):
    np.random.seed(seed)
    kwargs: dict = {}
    if topology == 'torus':
        kwargs.update(major_radius=1.0, minor_radius=1.0)
    elif topology == 'sphere':
        kwargs.update(radius=1.0)
    elif topology == 'helix':
        kwargs.update(radius=1.0, pitch=0.5, n_turns=4.0)
    dgp = make_dgp(topology, d=d, sigma=sigma, c0=np.zeros(d), **kwargs)
    return dgp.sample_with_gt(n)


@pytest.fixture(scope='session')
def circle_small():
    """N=500 circle in d=8, sigma=0.05, seed=0. Returns (X, gt)."""
    return _sample('circle', n=500, d=8, sigma=0.05, seed=0)


@pytest.fixture(scope='session')
def torus_small():
    """N=600 torus in d=8, sigma=0.05, seed=0. Returns (X, gt)."""
    return _sample('torus', n=600, d=8, sigma=0.05, seed=0)


@pytest.fixture(scope='session')
def sphere_small():
    """N=600 sphere in d=8, sigma=0.05, seed=0. Returns (X, gt)."""
    return _sample('sphere', n=600, d=8, sigma=0.05, seed=0)


@pytest.fixture(scope='session')
def helix_small():
    """N=500 open helix in d=8, sigma=0.05, seed=0. Returns (X, gt) where
    gt is shape (N, 1) raw arc-length parameter s in [0, L]."""
    return _sample('helix', n=500, d=8, sigma=0.05, seed=0)
