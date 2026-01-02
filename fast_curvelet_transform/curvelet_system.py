"""System for a simple implementation of the curvelet transform in jax. 

This is a "brute force" implementation of the curvelet transform from the
numpy code to fit a highly parallel version in jax. Here we create a bank or
filters for each scale and wedge and then perform the transform in a
vectorized manner. Here the filters are much larger than necessary and they
have a large memory footprint.

This approach is similar to the one used in the diffcurve library [1]. Although,
instead of using the implementation from Matlab coming form the curvelet toolbox
we use our own numpy implementation.

References: 

[1]: https://github.com/liutianlin0121/diffcurve

"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Any, Literal
from .curvelet import fdct_wrapping, ifdct_wrapping, CurveletOptions
from scipy import fft

Array = jax.Array


def np_fft2(x: np.ndarray) -> np.ndarray:
  """Computes the 2D FFT with correct shifting and normalization."""
  return fft.fftshift(fft.fft2(fft.ifftshift(x), norm='ortho'))


def jax_fft2(spatial_input: Array) -> Array:
  """Computes the 2D FFT with correct shifting and normalization in jax."""
  return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(spatial_input),
                                       norm='ortho'))


def jax_ifft2(frequency_input: Array) -> Array:
  """Computes the inverse 2D FFT with correct shifting and normalization"""
  return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(frequency_input),
                                        norm='ortho'))


@jax.tree_util.register_pytree_node_class
class CurveletSystem:
  """A JAX Pytree representing a curvelet system.

  This Pytree is used to store the curvelet waveforms and their dimensions.

  Attributes:
    waveforms: A JAX array containing all curvelet waveforms in the
      frequency domain.
    dimensions: A JAX array containing the dimensions of each curvelet
      coefficient block.
  """

  def __init__(self, waveforms: Array, dimensions: Array):
    self.waveforms = waveforms
    self.dimensions = dimensions

  def tree_flatten(self):
    children = (self.waveforms, self.dimensions)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

  def jax_fdct_2d(self, img: Array) -> Array:
    """2d fast discrete curvelet in jax

    Args:
      img: 2D array [m, n]

    Returns:
      coeffs: curvelet coefficients [num_curvelets, m, n]
    """
    x_freq = jax_fft2(img)
    conj_waveforms = jnp.conj(self.waveforms)
    coeffs = jax_ifft2(x_freq * conj_waveforms)
    return coeffs

  def jax_ifdct_2d(self, coeffs: Array) -> Array:
    """2d inverse fast discrete curvelet in jax.

    Args:
      coeffs: curvelet coefficients [num_curvelets, m, n]

    Returns:
      decomp: image decomposed in different scales and orientation in the
        curvelet basis [num_curvelets, m, n].
    """
    coeffs_freq = jax_fft2(coeffs)

    # Correctly compute support size for each wedge.
    support_size = jnp.prod(self.dimensions, axis=1)

    decomp = jax_ifft2(
      coeffs_freq * self.waveforms
    ) * jnp.expand_dims(support_size, [1, 2])

    return decomp

  def reconstruct(self, coeffs: Array) -> Array:
    """Reconstruct the image from curvelet coefficients.

    Args:
      coeffs: curvelet coefficients [num_curvelets, m, n]

    Returns:
      img: Reconstructed image [m, n]
    """
    decomp = self.jax_ifdct_2d(coeffs)
    return jnp.sum(decomp, axis=0)


def get_curvelet_system(
  img_length: int,
  img_width: int,
  options: CurveletOptions
) -> CurveletSystem:
  """Gets curvelet waveforms in the frequency domain.

  Args:
    img_length: The length of the image to be curvelet transformed.
    img_width: The width of the image to be curvelet transformed.
    options: CurveletOptions object containing transform parameters.

  Returns:
    A CurveletSystem object containing the waveforms and their dimensions.
  """
  # Create zero coefficients structure by running forward transform on 0 image.
  zeros = np.zeros((img_length, img_width), dtype=options.dtype)
  zero_coeffs = fdct_wrapping(
    zeros,
    is_real=options.is_real,
    finest=options.finest,
    nbscales=options.nbscales,
    nbangles_coarse=options.nbangles_coarse,
    dtype=options.dtype
  )

  all_scales_all_wedges_curvelet_coeffs = []
  curvelet_coeff_dim = []

  # Iterate through each scale and wedge.
  for scale_idx, curvelets_scale in enumerate(zero_coeffs):
    for wedge_idx, curvelet_wedge in enumerate(curvelets_scale):
      coeff_length, coeff_width = curvelet_wedge.shape
      curvelet_coeff_dim.append((coeff_length, coeff_width))

      coord_vert = int(coeff_length // 2)
      coord_horiz = int(coeff_width // 2)

      # Insert an impulse at the center of the current wedge.
      # We must be careful not to modify the original structure if we want to
      # reuse it, but here we can just set it and unset it.
      curvelet_wedge[coord_vert, coord_horiz] = 1.0

      # Perform inverse transform to get the spatial waveform.
      out = ifdct_wrapping(
        zero_coeffs,
        is_real=options.is_real,
        m_img=img_length,
        n_img=img_width,
        dtype=options.dtype
      )

      # Transform to frequency domain.
      out_freq = np_fft2(out)
      all_scales_all_wedges_curvelet_coeffs.append(out_freq)

      # Restore to zero for the next iteration.
      curvelet_wedge[coord_vert, coord_horiz] = 0.0

  # Convert results to JAX arrays for the CurveletSystem Pytree.
  waveforms = jnp.array(all_scales_all_wedges_curvelet_coeffs)
  dimensions = jnp.array(curvelet_coeff_dim)

  return CurveletSystem(waveforms, dimensions)
