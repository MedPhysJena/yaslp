from collections import namedtuple
from functools import partial
from typing import Iterable

import jax.numpy as np


def grid_basis(
    grid_shape: Iterable[int], reciprocal=False, normalise=False
) -> np.ndarray:
    if reciprocal:
        basis_vecs = [np.fft.fftfreq(size) for size in grid_shape]
    else:
        basis_vecs = [np.linspace(-1, 1, size) for size in grid_shape]

    grid = np.stack(np.meshgrid(*basis_vecs, indexing="ij"), axis=-1)

    if normalise:
        norm = np.linalg.norm(grid, axis=-1)
        grid /= np.where(norm == 0, np.inf, norm)[..., None]
    return grid


Ellipse = namedtuple("Ellipse", ["value", "radius", "center", "phi"])
ELLIPSES = [
    Ellipse(value=1, radius=(0.8, 0.9, 0.8), center=(0, 0, 0), phi=0),  # FG
    # The value of the following ellipses should be understood relative to the value
    # of the main, FG, ellipse defined above
    Ellipse(value=1, radius=(0.35, 0.12, 0.12), center=(-0.35, 0.1, 0), phi=1.9),  # A
    Ellipse(value=2, radius=(0.44, 0.15, 0.15), center=(0.35, 0.1, 0), phi=1.2),  # B
    Ellipse(value=3, radius=(0.12, 0.12, 0.40), center=(0, 0.5, 0), phi=0),  # C
    Ellipse(value=4, radius=(0.18, 0.18, 0.18), center=(0, -0.6, 0), phi=0),  # D
]


@partial(np.vectorize, signature="(n,n),(n)->(n)")
def _rotate(rot: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return rot.dot(vector)


def _embed_sphere(
    basis: np.ndarray, center: Iterable, radius: Iterable | float, phi: float
) -> np.ndarray:
    """
    Create a boolean array with an ellipsoid within the grid specified by the basis

    Parameters
    ----------
    basis : ndarray,
        ndarray of shape (n,) containing basis vectors
    center : array_like
        point in ndim space
    radius : array_like or float
        radius for a sphere, array_like of shape (n,) for an ellipsoid
    phi : float
        rotation angle (in rad) from (1, 0, 0) vector in xy plane

    Returns
    -------
    mask : ndarray
        boolean mask with the ellipsoid represented by True

    Examples
    --------
    >>> basis = grid_basis((8,8))
    >>> basis.shape
    (3,)
    >>> center = (-1,-1)
    >>> radius = (3,4)
    >>> mask = _embed_sphere(basis, center, radius, 0.)
    >>> mask.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    """
    rotation_matrix = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    basis_rotated = _rotate(rotation_matrix, basis - np.asarray(center))
    basis_normalized = basis_rotated / np.asarray(radius)
    return np.sum(basis_normalized**2, axis=-1) < 1


def shepp_logan(grid_shape: tuple, ellipses: list[Ellipse] = ELLIPSES) -> np.ndarray:
    grid = grid_basis(grid_shape)
    phantom = sum(
        [
            np.where(_embed_sphere(grid, e.center, e.radius, e.phi), e.value, 0)
            for e in ellipses
        ],
    )
    return phantom
