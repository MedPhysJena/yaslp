"""Test phantom generation."""

import pytest
import jax.numpy as jnp
from yaslp import phantom
from yaslp.utils import grid_basis


def test_gen_rotation_matrix_in_2d():
    """Test construction of a 2×2 rotation matrix."""
    phi = -jnp.pi / 2
    mat = phantom.gen_rotation_matrix_in_2d(phi)
    assert jnp.allclose(mat.T @ mat, jnp.eye(2))


def test_rotate():
    """Test application of a 2×2 rotation matrix to a stack of vectors."""
    phi = -jnp.pi / 2
    mat = phantom.gen_rotation_matrix_in_2d(phi)
    vectors = jnp.array([[1, 0], [0.5, 0.5], [0, 1]])
    vectors_rotated_actual = phantom._rotate(mat, vectors)
    vectors_rotated_expected = jnp.array([[0, 1], [-0.5, 0.5], [-1, 0]])
    assert jnp.allclose(
        vectors_rotated_actual, vectors_rotated_expected, atol=1e-4, rtol=1e-4
    )


def test_rotate_mismatched_dims():
    """Test that trying to rotate a stack of 3D vectors with 2×2 matrix fails."""
    phi = -jnp.pi / 2
    mat = phantom.gen_rotation_matrix_in_2d(phi)
    vectors = jnp.array([[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        phantom._rotate(mat, vectors)


@pytest.mark.parametrize(
    "grid_shape,radius",
    [
        ((5, 7, 10), (0.7, 0.5, 0.8)),
        ((5, 7, 10), 0.8),
        ((7, 10), (0.7, 0.5)),
        ((7, 10), 0.8),
    ],
)
def test_embed_sphere(grid_shape, radius):
    """Test that _embed_sphere can handle 3D and 3D inputs."""
    result = phantom._embed_sphere(
        grid_basis(grid_shape), (0.0,) * len(grid_shape), radius, -jnp.pi / 3
    )
    assert result.shape == grid_shape
    assert result.sum()


def test_shepp_logan():
    """Test that the default 2D phantom is in the centre of the default 3D phantom."""
    grid_shape_2d = (10, 7)
    result_2d = phantom.shepp_logan(grid_shape_2d)
    result_3d = phantom.shepp_logan(grid_shape_2d + (5,))
    assert jnp.allclose(result_2d, result_3d[..., 2])
