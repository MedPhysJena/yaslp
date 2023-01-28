from typing import Iterable

import jax.numpy as jnp


def grid_basis(
    grid_shape: Iterable[int], reciprocal=False, normalise=False
) -> jnp.ndarray:
    if reciprocal:
        basis_vecs = [jnp.fft.fftfreq(size) for size in grid_shape]
    else:
        basis_vecs = [jnp.linspace(-1, 1, size) for size in grid_shape]

    grid = jnp.stack(jnp.meshgrid(*basis_vecs, indexing="ij"), axis=-1)

    if normalise:
        norm = jnp.linalg.norm(grid, axis=-1)
        grid /= jnp.where(norm == 0, jnp.inf, norm)[..., None]
    return grid


def _perturb_all_but_one_component(grid: jnp.ndarray, axis: int) -> jnp.ndarray:
    perturb = grid.at[..., axis].set(0)
    return jnp.where(grid[..., axis][..., None] < 0, -perturb, perturb)


def perturb_vector_field(
    vectors: jnp.ndarray,
    phantom: jnp.ndarray,
    label_pivot_centres: dict[int, tuple],
    label_primary_axis: dict[int, int],
) -> jnp.ndarray:
    """Add linear perturbation to the vector field in selected ellipsoids

    Parameters
    ----------
    vectors : 4D jax.numpy.DeviceArray
    phantom : 3D jax.numpy.DeviceArray, integer-valued
    label_pivot_centres : map from the value of a perturbed ellipsoid to the desired
        pivoting point for the perturbation field. Often used:
        label_pivot_centres = {e.value + 1: e.center for e in ELLIPSES[1:4]}
    label_primary_axis : map from the value of a perturbed ellipsoid to the axis
        which is to be left unperturbed.
    """
    grid_shape = vectors.shape[:-1]
    grid = grid_basis(grid_shape)  # shape: grid_shape + (len(grid_shape),)

    pertubration = jnp.zeros_like(vectors)
    for phantom_value, roi_center in label_pivot_centres.items():
        perturb = _perturb_all_but_one_component(
            grid - jnp.array(roi_center), axis=label_primary_axis[phantom_value]
        )
        pertubration += jnp.where(phantom[..., None] == phantom_value, perturb, 0)

    return pertubration
