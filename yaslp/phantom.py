from collections import namedtuple
from functools import cached_property, partial
from typing import Iterable

import jax.numpy as jnp

from yaslp.utils import grid_basis

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


@partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def _rotate(rot: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return rot.dot(vector)


def _embed_sphere(
    basis: jnp.ndarray, center: Iterable, radius: Iterable | float, phi: float
) -> jnp.ndarray:
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
    rotation_matrix = jnp.array(
        [[jnp.cos(phi), jnp.sin(phi), 0], [-jnp.sin(phi), jnp.cos(phi), 0], [0, 0, 1]]
    )
    basis_rotated = _rotate(rotation_matrix, basis - jnp.asarray(center))
    basis_normalized = basis_rotated / jnp.asarray(radius)
    return jnp.sum(basis_normalized**2, axis=-1) < 1


class EllipsoidPhantom:
    def __init__(self, grid_shape: tuple, ellipses: list[Ellipse] = ELLIPSES):
        """Build an integer-valued phantom from a collection of ellipsoids

        (currently limited to 3D)"""
        assert (
            len(grid_shape) == 3
        ), "rotation_matrix in its present form expects a 3D space"
        self.grid = grid_basis(grid_shape)
        self.ellipses = ellipses
        self.masks = [
            _embed_sphere(self.grid, e.center, e.radius, e.phi) for e in ellipses
        ]
        self.n = len(ellipses)

    @cached_property
    def label_map(self):
        # why not just mask * e.value?
        return sum(
            [jnp.where(mask, e.value, 0) for mask, e in zip(self.masks, self.ellipses)]
        )

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.n} ellipses"


def shepp_logan(grid_shape: tuple, ellipses: list[Ellipse] = ELLIPSES) -> jnp.ndarray:
    return EllipsoidPhantom(grid_shape, ellipses).label_map
