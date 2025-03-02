from collections import namedtuple
from functools import cached_property, partial
from typing import Iterable

from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float, ArrayLike

from yaslp.utils import grid_basis, perturb_vector_field, safe_normalize

Ellipse = namedtuple("Ellipse", ["value", "radius", "center", "phi"])
ELLIPSES_IN_3D = [
    Ellipse(value=1, radius=(0.8, 0.9, 0.8), center=(0, 0, 0), phi=0),  # FG
    # The value of the following ellipses should be understood relative to the value
    # of the main, FG, ellipse defined above
    Ellipse(value=1, radius=(0.35, 0.12, 0.12), center=(-0.35, 0.1, 0), phi=1.9),  # A
    Ellipse(value=2, radius=(0.44, 0.15, 0.15), center=(0.35, 0.1, 0), phi=1.2),  # B
    Ellipse(value=3, radius=(0.12, 0.12, 0.40), center=(0, 0.5, 0), phi=0),  # C
    Ellipse(value=4, radius=(0.18, 0.18, 0.18), center=(0, -0.6, 0), phi=0),  # D
]
ELLIPSES_IN_2D = [
    Ellipse(value=1, radius=(0.8, 0.9), center=(0, 0), phi=0),  # FG
    # The value of the following ellipses should be understood relative to the value
    # of the main, FG, ellipse defined above
    Ellipse(value=1, radius=(0.35, 0.12), center=(-0.35, 0.1), phi=1.9),  # A
    Ellipse(value=2, radius=(0.44, 0.15), center=(0.35, 0.1), phi=1.2),  # B
    Ellipse(value=3, radius=(0.12, 0.12), center=(0, 0.5), phi=0),  # C
    Ellipse(value=4, radius=(0.18, 0.18), center=(0, -0.6), phi=0),  # D
]


def gen_rotation_matrix_in_2d(phi_rad: float) -> Float[Array, "2 2"]:
    """Generate a rotation matrix in 2D that implements rotation by phi radians."""
    return jnp.array(
        [[jnp.cos(phi_rad), jnp.sin(phi_rad)], [-jnp.sin(phi_rad), jnp.cos(phi_rad)]]
    )


@partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def _rotate(rot: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return rot.dot(vector)


def _embed_sphere(
    basis: Float[Array, "*batch ndim"],
    center: Float[ArrayLike, " ndim"],
    radius: Float[ArrayLike, " ndim"] | float,
    phi: float,
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
    ndim = basis.shape[-1]
    assert basis.ndim == ndim + 1, f"Unexpected shape of basis: {basis.shape}"
    if ndim not in [2, 3]:
        raise ValueError(f"_embed_sphere only supports 2D or 3D inputs, got {ndim}D")
    if jnp.size(center) != ndim:
        raise ValueError(f"{center=} is inconsistent with {basis.shape[-1]=}")
    if jnp.size(radius) not in [1, ndim]:
        raise ValueError(f"{radius=} is inconsistent with {basis.shape[-1]=}")

    rotation_matrix = gen_rotation_matrix_in_2d(phi)
    if ndim == 3:
        rotation_matrix = jnp.eye(3).at[:2, :2].set(rotation_matrix)
    basis_rotated = _rotate(rotation_matrix, basis - jnp.asarray(center))
    basis_normalized = basis_rotated / jnp.asarray(radius)
    return jnp.sum(basis_normalized**2, axis=-1) < 1


class EllipsoidPhantom:
    def __init__(self, grid_shape: tuple, ellipses: list[Ellipse] | None = None):
        """Build an integer-valued phantom from a collection of ellipsoids

        (currently limited to 3D)"""
        if len(grid_shape) not in [2, 3]:
            raise NotImplementedError("EllipsoidPhantom only supports 2D or 3D spaces.")

        self.grid = grid_basis(grid_shape)
        if ellipses is None and len(grid_shape) == 2:
            ellipses = ELLIPSES_IN_2D
        elif ellipses is None and len(grid_shape) == 3:
            ellipses = ELLIPSES_IN_3D
        assert ellipses is not None

        self.ellipses: list[Ellipse] = ellipses
        self.masks = [
            _embed_sphere(self.grid, e.center, e.radius, e.phi) for e in self.ellipses
        ]
        self.n = len(self.ellipses)

    @cached_property
    def label_map(self):
        # why not just mask * e.value?
        return sum(
            [jnp.where(mask, e.value, 0) for mask, e in zip(self.masks, self.ellipses)]
        )

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.n} ellipses"

    def disperse_vector_field(
        self, lut: jnp.ndarray, dispersion_factor=3.0, rng=None, concentration=1.0
    ):
        # FIX: pretty much the entire function below operates under presumption that
        # there is a single ellipsoid, overlapping with each FG ROI. Additionally,
        # the treatment of BG may not be general (and is certainly neither clear
        # nor uniform)

        # lut includes BG as 0th
        directional_labels = jnp.argwhere(
            jnp.linalg.norm(lut[1:], axis=-1) > 0
        ).flatten()
        vectors = lut[self.label_map]
        fg_roi_val2center = {
            self.ellipses[idx].value + 1: self.ellipses[idx].center
            for idx in directional_labels
        }
        vectors += dispersion_factor * perturb_vector_field(
            vectors,
            self.label_map,
            label_pivot_centres=fg_roi_val2center,
            label_primary_axis=jnp.argmax(lut, axis=1),
        )
        vectors_norm = jnp.linalg.norm(vectors, axis=-1)[..., None]
        vectors = jnp.nan_to_num(vectors / vectors_norm, posinf=0, neginf=0)
        if rng is not None:
            eps = random.normal(rng, shape=vectors.shape)
            vectors = safe_normalize(
                (vectors * concentration) + jnp.where(vectors_norm > 0, eps, 0)
            )
        return vectors


def shepp_logan(
    grid_shape: tuple, ellipses: list[Ellipse] | None = None
) -> jnp.ndarray:
    return EllipsoidPhantom(grid_shape, ellipses).label_map
