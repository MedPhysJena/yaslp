import warnings
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float


def grid_basis(
    grid_shape: tuple[int, ...],
    rfft=False,
    return_unit_vector=False,
    stacking_axis=-1,
    domain: Literal["-1:1", "int"] = "-1:1",
    reciprocal: bool = False,
    normalise: bool | None = None,
) -> Float[Array, "*batch ndim"]:
    if normalise is not None:
        warnings.warn(
            "'normalise' is depricated, set return_unit_vector=True", DeprecationWarning
        )
        return_unit_vector = normalise

    if reciprocal:
        basis_vecs = [jnp.fft.fftfreq(size) for size in grid_shape]
        if rfft:
            basis_vecs[-1] = jnp.fft.rfftfreq(grid_shape[-1])
        if domain == "int":
            basis_vecs = [vec / vec[1] for vec in basis_vecs]
    else:
        if rfft:
            warnings.warn("rfft=True is ignored if reciprocal=False")
        if domain == "-1:1":
            basis_vecs = [jnp.linspace(-1, 1, size) for size in grid_shape]
        elif domain == "int":
            basis_vecs = [jnp.arange(size) - size // 2 for size in grid_shape]
        else:
            raise ValueError(f"domain must be 'int' or '-1:1', got {domain}")

    grid = jnp.stack(jnp.meshgrid(*basis_vecs, indexing="ij"), axis=stacking_axis)

    if return_unit_vector:
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


def safe_normalize(x, *, p=2):
    """
    Safely project a vector onto the sphere wrt the ``p``-norm. This avoids the
    singularity at zero by mapping zero to the uniform unit vector proportional
    to ``[1, 1, ..., 1]``.

    :param numpy.ndarray x: A vector
    :param float p: The norm exponent, defaults to 2 i.e. the Euclidean norm.
    :returns: A normalized version ``x / ||x||_p``.
    :rtype: numpy.ndarray

    Copied verbatim from numpyro.distributions.util (Apache 2.0)
    """
    # TODO: refactor to suit my needs
    assert isinstance(p, (float, int))
    assert p >= 0
    norm = jnp.linalg.norm(x, p, axis=-1, keepdims=True)
    x = x / jnp.clip(norm, a_min=jnp.finfo(x).tiny)
    # Avoid the singularity.
    mask = jnp.all(x == 0, axis=-1, keepdims=True)
    x = jnp.where(mask, x.shape[-1] ** (-1 / p), x)
    return x
