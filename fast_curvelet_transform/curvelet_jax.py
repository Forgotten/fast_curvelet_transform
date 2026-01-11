"""JAX implementation of the Forward Fast Discrete Curvelet Transform (Wrapping)."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Literal
import functools
from . import curvelet_utils as cutils


@functools.partial(
  jax.jit,
  static_argnames=("is_real", "finest", "nbscales", "nbangles_coarse")
)
def fdct_wrapping_jax(
  x: jax.Array,
  is_real: bool = False,
  finest: Literal["wavelets", "curvelets"] = "curvelets",
  nbscales: int | None = None,
  nbangles_coarse: int = 16,
) -> List[List[jax.Array]]:
  """Fast Discrete Curvelet Transform via wedge wrapping (JAX).
  
  Args:
    x: Input image (2D JAX array).
    is_real: Whether the transform is real-valued.
    finest: Type of the finest scale ('wavelets' or 'curvelets').
    nbscales: Number of scales.
    nbangles_coarse: Number of angles at the coarsest level.
    
  Returns:
    A list of lists containing curvelet coefficients (j, l).
  """
  n1, n2 = x.shape
  
  if nbscales is None:
    nbscales = int(np.ceil(np.log2(min(n1, n2)) - 3))
    
  # Frequency domain transform
  # Using np.sqrt for normalization since size is concrete from static shape.
  xf = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x))) / np.sqrt(x.size)
  
  # Use NumPy version to get concrete counts for list construction.
  nbangles = cutils.get_nbangles(nbscales, nbangles_coarse, finest)
  c_coeffs: List[List[jax.Array | None]] = [
    [None] * nbangles[j] for j in range(nbscales)
  ]
  
  m1, m2 = n1 / 3.0, n2 / 3.0
  
  # Computing the finest scale.
  if finest == "curvelets":
    x_low = cutils.curvelet_finest_level_jax(n1, m1, n2, m2, xf)
    scales = range(nbscales - 1, 0, -1)
  elif finest == "wavelets":
    m1, m2 = m1 / 2.0, m2 / 2.0
    x_low, x_hi = cutils.wavelet_finest_level_jax(n1, m1, n2, m2, xf)

    coeffs_finest = jnp.fft.fftshift(
      jnp.fft.ifft2(jnp.fft.ifftshift(x_hi))
    ) * np.sqrt(x_hi.size)
    if is_real:
      coeffs_finest = jnp.real(coeffs_finest)
    c_coeffs[nbscales - 1][0] = coeffs_finest
    scales = range(nbscales - 2, 0, -1)
  else:
    raise ValueError(f"Invalid finest type: {finest}")

  for j in scales:
    m1, m2 = m1 / 2.0, m2 / 2.0
    x_low, x_hi = cutils.apply_digital_corona_filter_jax(x_low, n1, n2, m1, m2)
    
    l_idx = 0
    nbquadrants = 2 if is_real else 4 
    nbangles_perquad = nbangles[j] // 4
    
    for quadrant in range(1, nbquadrants + 1):
      mh, mv = (m2, m1) if quadrant % 2 == 1 else (m1, m2)
      # We use the NumPy version here because its outputs are used to 
      # calculate structural parameters (shapes, offsets) that must be 
      # concrete during JIT compilation.
      wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(
        nbangles_perquad, mh
      )
      
      # Prepare as tuples for static arguments.
      wedge_endpoints_tuple = tuple(wedge_endpoints)
      wedge_midpoints_tuple = tuple(wedge_midpoints)

      def process_wdata(wdata, q, angle_off):
        data_rot = jnp.rot90(wdata, -(q - 1))
        coeffs = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(data_rot))) * np.sqrt(data_rot.size)
        if not is_real:
          c_coeffs[j][angle_off] = coeffs
        else:
          c_coeffs[j][angle_off] = jnp.sqrt(2.0) * jnp.real(coeffs)
          c_coeffs[j][angle_off + nbangles[j] // 2] = jnp.sqrt(2.0) * jnp.imag(coeffs)

      # 1. Left corner wedge
      wdata_left = cutils.get_wrapped_filtered_data_left_jax(
        quadrant, nbangles_perquad, x_hi, mv, mh, wedge_endpoints_tuple, wedge_midpoints_tuple
      )
      process_wdata(wdata_left, quadrant, l_idx)
      l_idx += 1

      # 2. Regular wedges
      length_wedge = int(np.floor(4 * mv) - np.floor(mv))
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      
      for subl in range(2, nbangles_perquad):
        wdata_reg = cutils.compute_wrapped_data_jax(
          subl, quadrant, wedge_endpoints_tuple, wedge_midpoints_tuple, mh, mv, x_hi
        )
        process_wdata(wdata_reg, quadrant, l_idx)
        l_idx += 1

      # 3. Right corner wedge
      wdata_right = cutils.get_wrapped_filtered_data_right_jax(
        quadrant, nbangles_perquad, x_hi, f_r, mv, mh, wedge_endpoints_tuple, wedge_midpoints_tuple
      )
      process_wdata(wdata_right, quadrant, l_idx)
      l_idx += 1

      if quadrant < nbquadrants:
        x_hi = jnp.rot90(x_hi)

  # Coarsest scale
  c_coeffs[0][0] = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x_low))) * np.sqrt(x_low.size)
  if is_real:
    c_coeffs[0][0] = jnp.real(c_coeffs[0][0])
  
  return [[arr for arr in scale] for scale in c_coeffs]
