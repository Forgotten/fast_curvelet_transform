""" Utility functions for the Fast Discrete Curvelet Transform (Wrapping)
-------------------------------------------------------------------
This module provides lower-level utility functions for frequency domain 
partitioning, windowing, and data wrapping used by the FDCT.
"""
import numpy as np
from scipy import fft
from typing import List, Tuple, Literal, Any
import jax.numpy as jnp
import jax
import functools

def fdct_wrapping_window(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Creates the two halves of a C^inf compactly supported window.
  
  The window is designed to provide a smooth transition (partition of unity)
  in the frequency domain. It satisfies wl^2 + wr^2 = 1.
  
  Args:
    x: Input coordinate array in [0, 1].
    
  Returns:
    A tuple of (left_window, right_window).
  """
  x = np.asarray(x, dtype=np.float64)
  wr = np.zeros_like(x)
  wl = np.zeros_like(x)
  
  x_safe = np.copy(x)
  x_safe[np.abs(x_safe) < 2**-52] = 0
  
  mask = (x_safe > 0) & (x_safe < 1)
  wr[mask] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / x_safe[mask])))
  wr[x_safe <= 0] = 1
  
  wl[mask] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / (1 - x_safe[mask]))))
  wl[x_safe >= 1] = 1
  
  normalization = np.sqrt(wl**2 + wr**2)
  return wl / normalization, wr / normalization


@jax.jit
def fdct_wrapping_window_jax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Creates the two halves of a C^inf compactly supported window.
  
  The window is designed to provide a smooth transition (partition of unity)
  in the frequency domain. It satisfies wl^2 + wr^2 = 1.
  Here we have a jax version of the function, which should be faster when fully
  compiled.
  
  Args:
    x: Input coordinate array in [0, 1].
    
  Returns:
    A tuple of (left_window, right_window).
  """
  # Use float32 or float64 depending on environment settings.
  x = jnp.asarray(x)
  wr = jnp.zeros_like(x)
  wl = jnp.zeros_like(x)
  
  x_safe = jnp.copy(x)
  x_safe = jnp.where(jnp.abs(x_safe) < 2**-52, jnp.zeros_like(x_safe), x_safe)
  
  x_safe_mask = (x_safe > 0) & (x_safe < 1)
  x_compute_safe = jnp.where(x_safe_mask, x_safe, 0.5 * jnp.ones_like(x_safe))

  wr = jnp.where(
    x_safe_mask,
    jnp.exp(1 - 1 / (1 - jnp.exp(1 - 1 / x_compute_safe))),
    wr
  )
  wr = jnp.where(x_safe <= 0, jnp.ones_like(wr), wr)
  
  wl = jnp.where(
    x_safe_mask,
    jnp.exp(1 - 1 / (1 - jnp.exp(1 - 1 / (1 - x_compute_safe)))),
    wl
  )
  wl = jnp.where(x_safe >= 1, jnp.ones_like(wl), wl)

  normalization = jnp.sqrt(jnp.square(wl) + jnp.square(wr))
  return wl / normalization, wr / normalization


def get_lowpass_1d(
  n_size: int,
  m_size: float,
  wavelet_mode: bool = False
) -> np.ndarray:
  """Utility to generate 1D lowpass filters.
  
  Args:
    n_size: Signal size.
    m_size: Lowpass cutoff parameter.
    wavelet_mode: Whether to use wavelet mode.
    
  Returns:
    1D lowpass filter array.
  """
  if not wavelet_mode:
    win_len = int(np.floor(2 * m_size) - np.floor(m_size) - 1 - (n_size % 3 == 0))
  else:
    win_len = int(np.floor(2 * m_size) - np.floor(m_size) - 1)
    
  coord = np.linspace(0, 1, win_len + 1)
  wl, wr = fdct_wrapping_window(coord)
  lowpass = np.concatenate([wl, np.ones(2 * int(np.floor(m_size)) + 1), wr])
  
  if not wavelet_mode and n_size % 3 == 0:
    lowpass = np.concatenate([[0], lowpass, [0]])
  return lowpass


def get_low_high_pass_2d(
  n1: int,
  m1: float,
  n2: int,
  m2: float,
  wavelet_mode: bool = False
) -> tuple[np.ndarray, np.ndarray]:
  """Utility to generate 2D lowpass and highpass filters.
  
  Args:
    n1: Height size.
    m1: Height lowpass cutoff.
    n2: Width size.
    m2: Width lowpass cutoff.
    wavelet_mode: Whether to use wavelet mode.
    
  Returns:
    A tuple of (lowpass_2d, highpass_2d) arrays.
  """
  lowpass = np.outer(
    get_lowpass_1d(n1, m1, wavelet_mode),
    get_lowpass_1d(n2, m2, wavelet_mode)
  )
  highpass = np.sqrt(np.maximum(0, 1 - lowpass**2))
  return lowpass, highpass


@functools.partial(jax.jit, static_argnames=("n_size", "m_size", "wavelet_mode"))
def get_lowpass_1d_jax(
  n_size: int,
  m_size: float,
  wavelet_mode: bool = False
) -> jax.Array:
  """Utility to generate 1D lowpass filters in JAX.
  
  Args:
    n_size: Signal size.
    m_size: Lowpass cutoff parameter.
    wavelet_mode: Whether to use wavelet mode.
    
  Returns:
    1D lowpass filter array.
  """
  if not wavelet_mode:
    win_len = int(np.floor(2 * m_size) - np.floor(m_size) - 1 - (n_size % 3 == 0))
  else:
    win_len = int(np.floor(2 * m_size) - np.floor(m_size) - 1)
    
  coord = jnp.linspace(0, 1, win_len + 1)
  wl, wr = fdct_wrapping_window_jax(coord)
  lowpass = jnp.concatenate([wl, jnp.ones(2 * int(np.floor(m_size)) + 1), wr])
  
  if not wavelet_mode and n_size % 3 == 0:
    lowpass = jnp.concatenate([jnp.array([0.0]), lowpass, jnp.array([0.0])])
  return lowpass

@functools.partial(jax.jit, static_argnames=("n1", "m1", "n2", "m2", "wavelet_mode"))
def get_low_high_pass_2d_jax(
  n1: int,
  m1: float,
  n2: int,
  m2: float,
  wavelet_mode: bool = False
) -> tuple[jax.Array, jax.Array]:
  """Utility to generate 2D lowpass and highpass filters in JAX.
  
  Args:
    n1: Height size.
    m1: Height lowpass cutoff.
    n2: Width size.
    m2: Width lowpass cutoff.
    wavelet_mode: Whether to use wavelet mode.
    
  Returns:
    A tuple of (lowpass_2d, highpass_2d) arrays.
  """
  lowpass = jnp.outer(
    get_lowpass_1d_jax(n1, m1, wavelet_mode),
    get_lowpass_1d_jax(n2, m2, wavelet_mode)
  )
  highpass = jnp.sqrt(jnp.maximum(0, 1 - jnp.square(lowpass)))
  return lowpass, highpass


def get_nbangles(
  nbscales: int,
  nbangles_coarse: int,
  finest: Literal["wavelets", "curvelets"]
) -> np.ndarray:
  """Calculates the number of angles at each scale.
  
  Args:
    nbscales: Number of scales.
    nbangles_coarse: Number of angles at the coarsest level.
    finest: Finest scale type ("curvelets" or "wavelets").
    
  Returns:
    Array containing number of angles per scale.
  """
  nbangles = np.zeros(nbscales, dtype=int)
  nbangles[0] = 1
  for j in range(1, nbscales):
    nbangles[j] = nbangles_coarse * 2**int(np.ceil((j - 1) / 2))
  
  if finest == "wavelets":
    nbangles[nbscales - 1] = 1
    
  return nbangles


@functools.partial(
  jax.jit,
  static_argnames=("nbscales", "nbangles_coarse", "finest")
)
def get_nbangles_jax(
  nbscales: int,
  nbangles_coarse: int,
  finest: Literal["wavelets", "curvelets"]
) -> jax.Array:
  """JAX version of get_nbangles.
  
  Args:
    nbscales: Number of scales.
    nbangles_coarse: Number of angles at the coarsest level.
    finest: Finest scale type ("curvelets" or "wavelets").
    
  Returns:
    Array containing number of angles per scale.
  """
  nbangles = jnp.zeros(nbscales, dtype=jnp.int32)
  nbangles = nbangles.at[0].set(1)
  
  if nbscales > 1:
    indices = jnp.arange(1, nbscales)
    values = nbangles_coarse * 2**(jnp.ceil((indices - 1) / 2).astype(jnp.int32))
    nbangles = nbangles.at[indices].set(values)
  
  if finest == "wavelets" and nbscales > 0:
    nbangles = nbangles.at[nbscales - 1].set(1)
    
  return nbangles


def curvelet_finest_level(
    n1: int, m1: float, n2: int, m2: float, xf: np.ndarray) -> np.ndarray: 
  big_n1 = 2 * int(np.floor(2 * m1)) + 1
  big_n2 = 2 * int(np.floor(2 * m2)) + 1
  idx1 = np.mod(
    np.floor(n1/2) - np.floor(2 * m1) + np.arange(big_n1), n1
  ).astype(int)
  idx2 = np.mod(
    np.floor(n2/2) - np.floor(2 * m2) + np.arange(big_n2), n2
  ).astype(int)
  lowpass, _ = get_low_high_pass_2d(n1, m1, n2, m2)
  x_low = xf[np.ix_(idx1, idx2)] * lowpass
  return x_low


@functools.partial(jax.jit, static_argnames=("n1", "m1", "n2", "m2"))
def curvelet_finest_level_jax(
  n1: int,
  m1: float,
  n2: int,
  m2: float,
  xf: jax.Array,
) -> jax.Array:
  """JAX version of curvelet_finest_level."""
  big_n1 = 2 * int(np.floor(2 * m1)) + 1
  big_n2 = 2 * int(np.floor(2 * m2)) + 1
  idx1 = jnp.mod(
    jnp.floor(n1 / 2) - np.floor(2 * m1) + jnp.arange(big_n1), n1
  ).astype(jnp.int32)
  idx2 = jnp.mod(
    jnp.floor(n2 / 2) - np.floor(2 * m2) + jnp.arange(big_n2), n2
  ).astype(jnp.int32)
  lowpass, _ = get_low_high_pass_2d_jax(n1, m1, n2, m2)
  x_low = xf[jnp.ix_(idx1, idx2)] * lowpass
  return x_low


def wavelet_finest_level(
    n1: int, m1: float, n2: int, m2: float, xf: np.ndarray) -> np.ndarray:
  """Utility to generate the high and lowpass filters for the finest scale.

  Here we consider the wavelet mode, where a wavelet is used a the finest scale. 

  Args:
    n1: Height size.
    m1: Height lowpass cutoff.
    n2: Width size.
    m2: Width lowpass cutoff.
    xf: Image in frequency domain.
    
  Returns:
    A tuple of (lowpass_2d, highpass_2d) arrays.
  """

  lowpass, hipass = get_low_high_pass_2d(n1, m1, n2, m2, True)
  idx1 = np.arange(
    -int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1
  ) + int(np.ceil((n1 + 1) / 2)) - 1
  idx2 = np.arange(
    -int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1
  ) + int(np.ceil((n2 + 1) / 2)) - 1
  
  x_low = xf[np.ix_(idx1, idx2)] * lowpass
  x_hi = xf.copy()
  x_hi[np.ix_(idx1, idx2)] *= hipass
  
  return x_low, x_hi


@functools.partial(jax.jit, static_argnames=("n1", "m1", "n2", "m2"))
def wavelet_finest_level_jax(
  n1: int,
  m1: float,
  n2: int,
  m2: float,
  xf: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """JAX version of wavelet_finest_level."""
  lowpass, hipass = get_low_high_pass_2d_jax(n1, m1, n2, m2, True)
  idx1 = jnp.arange(
    -int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1
  ) + int(np.ceil((n1 + 1) / 2)) - 1
  idx2 = jnp.arange(
    -int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1
  ) + int(np.ceil((n2 + 1) / 2)) - 1
  
  x_low = xf[jnp.ix_(idx1, idx2)] * lowpass
  x_hi = xf.at[jnp.ix_(idx1, idx2)].multiply(hipass)
  
  return x_low, x_hi


@functools.partial(
  jax.jit,
  static_argnames=("nbangles_perquad", "m_horiz")
)
def get_wedge_ticks_jax(nbangles_perquad: int, m_horiz: float) -> jax.Array:
  """JAX version of get_wedge_ticks.
  
  Args:
    nbangles_perquad: Number of angles per quadrant.
    m_horiz: Horizontal size parameter.
    
  Returns:
    Wedge ticks array.
  """
  ticks = jnp.linspace(0, 0.5, nbangles_perquad + 1)
  wedge_ticks_left = jnp.round(
    ticks * 2 * jnp.floor(4 * m_horiz) + 1
  ).astype(jnp.int32)
  wedge_ticks_right = (
    2 * jnp.floor(4 * m_horiz) + 2 - wedge_ticks_left
  ).astype(jnp.int32)
  
  if nbangles_perquad % 2 == 1:
    wedge_ticks = jnp.concatenate([wedge_ticks_left, wedge_ticks_right[::-1]])
  else:
    wedge_ticks = jnp.concatenate([wedge_ticks_left, wedge_ticks_right[-2::-1]])
  
  return wedge_ticks


@functools.partial(
  jax.jit,
  static_argnames=("nbangles_perquad", "m_horiz")
)
def get_wedge_end_mid_points_jax(
  nbangles_perquad: int,
  m_horiz: float
) -> tuple[jax.Array, jax.Array]:
  """JAX version of get_wedge_end_mid_points.
  
  Args:
    nbangles_perquad: Number of angles per quadrant.
    m_horiz: Horizontal size parameter.
    
  Returns:
    Wedge endpoints and midpoints arrays.
  """
  wedge_ticks = get_wedge_ticks_jax(nbangles_perquad, m_horiz)
  wedge_endpoints = wedge_ticks[1::2]
  wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2.0
  return wedge_endpoints, wedge_midpoints


def get_wedge_ticks(nbangles_perquad: int, m_horiz: float) -> np.ndarray:
  """Calculates wedge ticks for a quadrant.
  
  Args:
    nbangles_perquad: Number of angles per quadrant.
    m_horiz: Horizontal size parameter.
    
  Returns:
    Wedge ticks array.
  """
  ticks = np.linspace(0, 0.5, nbangles_perquad + 1)
  wedge_ticks_left = np.round(ticks * 2 * np.floor(4 * m_horiz) + 1).astype(int)
  wedge_ticks_right = 2 * np.floor(4 * m_horiz) + 2 - wedge_ticks_left
  
  if nbangles_perquad % 2 == 1:
    wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[::-1]])
  else:
    wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[-2::-1]])
  
  return wedge_ticks


def get_wedge_end_mid_points(nbangles_perquad: int, m_horiz: float) -> np.ndarray:
  """Calculates wedge endpoints and midpoints for a quadrant.
  
  Args:
    nbangles_perquad: Number of angles per quadrant.
    m_horiz: Horizontal size parameter.
    
  Returns:
    Wedge endpoints and midpoints arrays.
  """
  wedge_ticks = get_wedge_ticks(nbangles_perquad, m_horiz)
  wedge_endpoints = wedge_ticks[1::2]
  wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2.0
  return wedge_endpoints, wedge_midpoints


def wrap_data(
  length_wedge: int,
  ww: int,
  l_l: np.ndarray,
  f_c: int,
  x_hi: np.ndarray,
  f_r: int,
  *,
  type_wedge: Literal["regular", "left", "right"] = "regular",
  mh: int | None = None,
  dtype_coord: Any = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Wraps data for a given wedge in one of the quadrants.
  
  Args:
    length_wedge: Length of the wedge.
    ww: Width of the wedge.
    l_l: Low-pass filter.
    f_c: Center frequency.
    x_hi: High-pass filtered image.
    f_r: Reference frequency.
    type_wedge: Type of wedge ("regular", "left", or "right").
    mh: Number of rows in the image. Only needed for the right corner wedge.
    dtype_coord: Data type for the coordinate arrays w_xx and w_yy.
    
  Returns:
    Tuple of (wdata, w_xx, w_yy).
  """
  if type_wedge == "right" and mh is None:
    raise ValueError("mh must be provided for right corner wedge.")
    
  wdata = np.zeros((length_wedge, ww), dtype=np.complex128)
  w_xx, w_yy = np.zeros_like(wdata, dtype=dtype_coord), np.zeros_like(wdata, dtype=dtype_coord)
  rows_wedge = np.arange(1, length_wedge + 1)
  
  for r_idx, r in enumerate(rows_wedge):
    # Perform the actual 'wrapping' by calculating the corresponding column 
    # indices in the frequency plane.
    cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
    # 'adm' handles the coordinate mapping including possible shearing.
    # 'mask' is used to handle the case where the column indices are out of bounds.
    if type_wedge == "left":
      adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
      mask = cols > 0
    elif type_wedge == "right":
      adm = np.round(
        0.5 * (
          cols + 2 * np.floor(4 * mh) + 1 - 
          np.abs(cols - (2 * np.floor(4 * mh) + 1))
        )
      ).astype(int)
      mask = cols <= 2 * np.floor(4 * mh) + 1
    elif type_wedge == "regular": 
      adm = cols
      mask = np.ones_like(cols, dtype=bool)
    else:
      raise ValueError("Invalid type_wedge value.")
      
    nr = 1 + np.mod(r - f_r, length_wedge)
    wdata[nr - 1, :] = x_hi[r - 1, adm - 1] * mask
    w_xx[nr - 1, :] = adm
    w_yy[nr - 1, :] = r
    
  return wdata, w_xx, w_yy


@functools.partial(
  jax.jit,
  static_argnames=("length_wedge", "ww", "f_c", "f_r", "type_wedge", "mh", "dtype_coord")
)
def wrap_data_jax(
  length_wedge: int,
  ww: int,
  l_l: jax.Array,
  f_c: int,
  x_hi: jax.Array,
  f_r: int,
  *,
  type_wedge: Literal["regular", "left", "right"] = "regular",
  mh: int | None = None,
  dtype_coord: Any = jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Wraps data for a given wedge in one of the quadrants in JAX.
  
  Args:
    length_wedge: Length of the wedge.
    ww: Width of the wedge.
    l_l: Low-pass filter.
    f_c: Center frequency.
    x_hi: High-pass filtered image.
    f_r: Reference frequency.
    type_wedge: Type of wedge ("regular", "left", or "right").
    mh: Number of rows in the image. Only needed for the right corner wedge.
    dtype_coord: Data type for the coordinate arrays w_xx and w_yy.
    
  Returns:
    Tuple of (wdata, w_xx, w_yy).
  """
  if type_wedge == "right" and mh is None:
    raise ValueError("mh must be provided for right corner wedge.")

  rows_wedge = jnp.arange(1, length_wedge + 1)
  cols = l_l[:, None] + jnp.mod(jnp.arange(ww)[None, :] - (l_l[:, None] - f_c), ww)

  if type_wedge == "left":
    adm = jnp.round(0.5 * (cols + 1 + jnp.abs(cols - 1))).astype(jnp.int32)
    mask = cols > 0
  elif type_wedge == "right":
    val = 2 * jnp.floor(4 * mh) + 1
    adm = jnp.round(0.5 * (cols + val - jnp.abs(cols - val))).astype(jnp.int32)
    mask = cols <= val
  else:  # regular
    adm = cols.astype(jnp.int32)
    mask = jnp.ones_like(cols, dtype=jnp.bool_)

  # Compute data at original row positions
  row_indices = jnp.arange(length_wedge)
  data_at_orig_rows = x_hi[row_indices[:, None], adm - 1] * mask
  
  # Target row cyclic shift instead of at[].set()
  # Use jnp.roll for row permutation (cyclic shift)
  shift = 1 - f_r
  wdata = jnp.roll(data_at_orig_rows, shift, axis=0)
  w_xx = jnp.roll(adm.astype(dtype_coord), shift, axis=0)
  w_yy_shifted = jnp.roll(rows_wedge[:, None].astype(dtype_coord), shift, axis=0)
  w_yy = jnp.broadcast_to(w_yy_shifted, (length_wedge, ww))

  return wdata, w_xx, w_yy


def apply_digital_corona_filter(
  x_low: np.ndarray,
  n1: int,
  n2: int,
  m1: float,
  m2: float
) -> tuple[np.ndarray, np.ndarray]:
  """Applies a digital corona filter to extract a single scale."""
  lowpass_next, hipass = get_low_high_pass_2d(n1, m1, n2, m2, True)
  x_hi = x_low.copy()
  idx1 = np.arange(
    -int(np.floor(2*m1)), int(np.floor(2*m1)) + 1
  ) + int(np.floor(4 * m1))
  idx2 = np.arange(
    -int(np.floor(2*m2)), int(np.floor(2*m2)) + 1
  ) + int(np.floor(4 * m2))
  x_low_new = x_low[np.ix_(idx1, idx2)]
  x_hi[np.ix_(idx1, idx2)] = x_low_new * hipass
  x_low = x_low_new * lowpass_next
  return x_low, x_hi


@functools.partial(
  jax.jit,
  static_argnames=("n1", "n2", "m1", "m2")
)
def apply_digital_corona_filter_jax(
  x_low: jax.Array,
  n1: int,
  n2: int,
  m1: float,
  m2: float
) -> tuple[jax.Array, jax.Array]:
  """JAX version of apply_digital_corona_filter.
  
  Applies a digital corona filter to extract a single scale in JAX.
  """
  lowpass_next, hipass = get_low_high_pass_2d_jax(n1, m1, n2, m2, True)
  x_hi = x_low
  
  idx1 = jnp.arange(
    -int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1
  ) + int(np.floor(4 * m1))
  idx2 = jnp.arange(
    -int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1
  ) + int(np.floor(4 * m2))
  
  x_low_new = x_low[jnp.ix_(idx1, idx2)]
  x_hi = x_hi.at[jnp.ix_(idx1, idx2)].multiply(hipass)
  x_low_new = x_low_new * lowpass_next
  
  return x_low_new, x_hi

def compute_wrapped_data(
  subl: int,
  quadrant: int,
  wedge_endpoints: np.ndarray,
  wedge_midpoints: np.ndarray,
  mh: float,
  mv: float,
  x_hi: np.ndarray,
) -> np.ndarray:
  """Computes wrapped data for a given wedge in one of the quadrants."""
  
  # Computing coordinates for wrapping.
  
  length_wedge = int(np.floor(4 * mv) - np.floor(mv))
  rows_wedge = np.arange(1, length_wedge + 1)
  f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
  
  ww = int(wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1)
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[subl - 1]) / np.floor(4 * mv)
  l_l = np.round(
    wedge_endpoints[subl - 2] + sl_w * (rows_wedge - 1)
  ).astype(int)
  f_c = int(
    np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
    ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0)
  )
  
  # Wrapping the data.
  wdata, w_xx, w_yy = wrap_data(
    length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge="regular"
  )
  
  # Computes the slopes of the wedges.
  slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 2]) / np.floor(4 * mv)
  slope_wedge_right = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 1]) / np.floor(4 * mv)
  
  # Compute coordinates.
  c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
          (w_xx - wedge_midpoints[subl - 2] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
          (w_xx - wedge_midpoints[subl - 1] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  
  # Compute window functions.
  wl_l, _ = fdct_wrapping_window(c_l)
  _, wr_r = fdct_wrapping_window(c_r)
  
  return wdata * wl_l * wr_r

# TODO: only use jit when the function is called.
@functools.partial(
  jax.jit,
  static_argnames=("subl", "quadrant", "mh", "mv", "wedge_endpoints", "wedge_midpoints", "dtype_coord")
)
def compute_wrapped_data_jax(
  subl: int,
  quadrant: int,
  wedge_endpoints: jax.Array,
  wedge_midpoints: jax.Array,
  mh: float,
  mv: float,
  x_hi: jax.Array,
  dtype_coord: Any = jnp.float32,
) -> jax.Array:
  """Computes wrapped data for a given wedge in one of the quadrants in JAX."""
  
  length_wedge = int(np.floor(4 * mv) - np.floor(mv))
  rows_wedge = jnp.arange(1, length_wedge + 1)
  
  # f_r calculation
  f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
  
  ww = int(wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1)
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[subl - 1]) / np.floor(4 * mv)
  l_l = jnp.round(
    wedge_endpoints[subl - 2] + sl_w * (rows_wedge - 1)
  ).astype(jnp.int32)
  
  f_c = int(
    np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
    ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0)
  )
  
  # Wrapping the data.
  wdata, w_xx, w_yy = wrap_data_jax(
    length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge="regular", dtype_coord=dtype_coord
  )
  
  # Compute window functions using the centralized helper.
  wl_l, wr_r = get_wedge_window_filters_jax_from_coords(
    w_xx, w_yy, subl, l_l, f_c, f_r, ww, mv, mh, wedge_endpoints, wedge_midpoints, type_wedge="regular"
  )
  
  return wdata * wl_l * wr_r


@functools.partial(
  jax.jit,
  static_argnames=("length_wedge", "ww", "f_r", "f_c", "dtype_coord")
)
def compute_regular_wedges_vectorized_jax(
  l_ls: jax.Array,
  mh_floor: float,
  mv_floor: float,
  wedge_endpoints: jax.Array,
  wedge_midpoints: jax.Array,
  x_hi: jax.Array,
  length_wedge: int,
  ww: int,
  f_r: int,
  f_c: int,
  dtype_coord: Any = jnp.float32,
) -> jax.Array:
  """Computes all regular wedges in a quadrant using vmap."""
  
  # l_ls size: (nbangles_reg, length_wedge)
  # wedge_endpoints size: (nbangles_perquad + 1)
  # wedge_midpoints size: (nbangles_perquad + 1)
  
  def single_wedge_logic(l_l, ep_left_2, ep_left_1, ep_right, mp_left, mp_right, subl_idx):
    # Wrapping
    wdata, w_xx, w_yy = wrap_data_jax(
      length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge="regular", dtype_coord=dtype_coord
    )
    
    # Windowing using the centralized helper
    wl_l, wr_r = get_wedge_window_filters_jax_from_coords(
      w_xx, w_yy, subl_idx, l_l, f_c, f_r, ww, mv_floor/4.0, mh_floor/4.0, wedge_endpoints, wedge_midpoints, type_wedge="regular"
    )
    
    return wdata * wl_l * wr_r

  # nbangles_reg is inferred from l_ls.shape[0]
  subl_indices = jnp.arange(2, l_ls.shape[0] + 2)
  
  res = jax.vmap(single_wedge_logic)(
    l_ls,
    wedge_endpoints[subl_indices - 2], # ep_left_2
    wedge_endpoints[subl_indices - 1], # ep_left_1
    wedge_endpoints[subl_indices],     # ep_right
    wedge_midpoints[subl_indices - 2], # mp_left
    wedge_midpoints[subl_indices - 1], # mp_right
    subl_indices                       # subl_idx
  )
  
  return res



def aggregate_from_wrapped_data(
  wdata: np.ndarray,
  xj: np.ndarray,
  l_l: np.ndarray,
  f_c: int,
  f_r: int,
  lcw: int,
  ww: int,
  type_wedge: Literal["left", "right", "regular"] = "regular",
  mh: float | None = None
) -> np.ndarray:
  """Aggregates the wrapped data into the x_j array.
  
  Args:
    wdata: Wrapped data array.
    xj: Array to store the aggregated data.
    l_l: Array of left limits.
    f_c: Coarsest frequency.
    f_r: Reference frequency.
    lcw: Number of concentric squares.
    ww: Number of wedges.
    type_wedge: Type of wedge.
    mh: Horizontal size parameter (needed for "right" type).
  
  Returns:
    Aggregated data array.
  """
  rows_wedge = np.arange(1, lcw + 1)
  
  for r_idx, r in enumerate(rows_wedge):
    cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
    if type_wedge == "left":
      adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
    elif type_wedge == "right":
      if mh is None:
        raise ValueError("mh must be provided for right corner wedge.")
      val = 2 * np.floor(4 * mh) + 1
      adm = np.round(0.5 * (cols + val - np.abs(cols - val))).astype(int)
    elif type_wedge == "regular":
      adm = cols
    nr = 1 + np.mod(r - f_r, lcw)
    xj[r - 1, adm - 1] += wdata[nr - 1, :]
    
  return xj


@functools.partial(
  jax.jit,
  static_argnames=("f_c", "f_r", "lcw", "ww", "type_wedge", "mh")
)
def aggregate_from_wrapped_data_jax(
  wdata: jax.Array,
  xj: jax.Array,
  l_l: jax.Array,
  f_c: int,
  f_r: int,
  lcw: int,
  ww: int,
  type_wedge: Literal["left", "right", "regular"] = "regular",
  mh: float | None = None
) -> jax.Array:
  """Aggregates the wrapped data into the x_j array in JAX.
  
  Args:
    wdata: Wrapped data array.
    xj: Array to store the aggregated data.
    l_l: Array of left limits.
    f_c: Coarsest frequency.
    f_r: Reference frequency.
    lcw: Number of concentric squares.
    ww: Number of wedges.
    type_wedge: Type of wedge.
    mh: Horizontal size parameter (needed for "right" type).
  
  Returns:
    Aggregated data array.
  """
  rows_wedge = jnp.arange(1, lcw + 1)
  cols = l_l[:, None] + jnp.mod(jnp.arange(ww)[None, :] - (l_l[:, None] - f_c), ww)
  
  if type_wedge == "left":
    adm = jnp.round(0.5 * (cols + 1 + jnp.abs(cols - 1))).astype(jnp.int32)
  elif type_wedge == "right":
    if mh is None:
      raise ValueError("mh must be provided for right corner wedge.")
    val = 2 * jnp.floor(4 * mh) + 1
    adm = jnp.round(0.5 * (cols + val - jnp.abs(cols - val))).astype(jnp.int32)
  elif type_wedge == "regular":
    adm = cols.astype(jnp.int32)
  else:
    raise NotImplementedError(f"type_wedge {type_wedge} not implemented for aggregate_from_wrapped_data_jax")

  # row translation
  target_rows = rows_wedge # 0-indexed r-1 in original is row_wedge-1
  
  # Let's use row_indices 0..lcw-1
  row_inds = jnp.arange(lcw)
  nr_minus_1 = jnp.mod(row_inds + 1 - f_r, lcw)
  
  vals_to_add = xj[row_inds[:, None], adm - 1] + wdata[nr_minus_1, :]
  return xj.at[row_inds[:, None], adm - 1].set(vals_to_add)


def get_wedge_window_filters(
  length_wedge: int,
  subl: int,
  l_l: np.ndarray,
  f_c: int,
  f_r: int,
  ww: int,
  mv: float,
  mh: float,
  wedge_endpoints: np.ndarray,
  wedge_midpoints: np.ndarray,
  type_wedge: Literal["left", "right", "regular"] = "regular"
) -> tuple[np.ndarray, np.ndarray]:
  """Computes the window functions for the given wedge."""
  w_xx, w_yy = np.zeros((length_wedge, ww)), np.zeros((length_wedge, ww))
  
  rows_wedge = np.arange(1, length_wedge + 1)
  for r_idx, r in enumerate(rows_wedge):
    cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
    w_xx[1+np.mod(r-f_r, length_wedge)-1, :] = cols
    w_yy[1+np.mod(r-f_r, length_wedge)-1, :] = r
    
  slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 2]) / np.floor(4 * mv)
  c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
          (w_xx - wedge_midpoints[subl - 2] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  slope_wedge_right = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 1]) / np.floor(4 * mv)
  c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
          (w_xx - wedge_midpoints[subl - 1] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  wl_l, _ = fdct_wrapping_window(c_l)
  _, wr_r = fdct_wrapping_window(c_r)
    
  return wl_l, wr_r


@functools.partial(
  jax.jit,
  static_argnames=(
    "f_c", "f_r", "ww",
    "type_wedge"
  )
)
def get_wedge_window_filters_jax_from_coords(
  w_xx: jax.Array,
  w_yy: jax.Array,
  subl: int | jax.Array,
  l_l: jax.Array,
  f_c: int,
  f_r: int,
  ww: int,
  mv: float,
  mh: float,
  wedge_endpoints: jax.Array | tuple[float, ...],
  wedge_midpoints: jax.Array | tuple[float, ...],
  type_wedge: Literal["left", "right", "regular"] = "regular"
) -> tuple[jax.Array, jax.Array]:
  """Computes the window functions for the given wedge in JAX using precomputed coordinates."""
  
  wedge_endpoints = jnp.array(wedge_endpoints)
  wedge_midpoints = jnp.array(wedge_midpoints)
  
  mv_floor = jnp.floor(4 * mv)
  mh_floor = jnp.floor(4 * mh)
  
  if type_wedge == "regular":
    slope_wedge_left = (mh_floor + 1 - wedge_midpoints[subl - 2]) / mv_floor
    c_l = 0.5 + mv_floor / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
            (w_xx - wedge_midpoints[subl - 2] - slope_wedge_left * (w_yy - 1)) / (mv_floor + 1 - w_yy)
    slope_wedge_right = (mh_floor + 1 - wedge_midpoints[subl - 1]) / mv_floor
    c_r = 0.5 + mv_floor / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
            (w_xx - wedge_midpoints[subl - 1] - slope_wedge_right * (w_yy - 1)) / (mv_floor + 1 - w_yy)
  elif type_wedge == "left":
    fwev = jnp.round(2 * mv_floor / (2 * (len(wedge_endpoints)-1)) + 1)
    slope_wedge_right = (mh_floor + 1 - wedge_midpoints[0]) / mv_floor
    c_r = 0.5 + mv_floor / (wedge_endpoints[1] - wedge_endpoints[0]) * \
            (w_xx - wedge_midpoints[0] - slope_wedge_right * (w_yy - 1)) / (mv_floor + 1 - w_yy)
    c2_const = 1.0 / (1.0 / (2 * mh_floor / (wedge_endpoints[0] - 1) - 1) + \
            1.0 / (2 * mv_floor / (fwev - 1) - 1))
    c1_const = c2_const / (2 * mv_floor / (fwev - 1) - 1)
    mask_c = ((w_xx - 1) / mh_floor + (w_yy - 1) / mv_floor == 2)
    w_xx_mod = jnp.where(mask_c, w_xx + 1, w_xx)
    c_l = c1_const + c2_const * ((w_xx_mod - 1) / mh_floor - (w_yy - 1) / mv_floor) / \
            (2 - ((w_xx_mod - 1) / mh_floor + (w_yy - 1) / mv_floor))
  elif type_wedge == "right":
    fwev = jnp.round(2 * mv_floor / (2 * (len(wedge_endpoints)-1)) + 1)
    slope_wedge_left = (mh_floor + 1 - wedge_midpoints[-1]) / mv_floor
    c_l = 0.5 + mv_floor / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
           (w_xx - wedge_midpoints[-1] - slope_wedge_left * (w_yy - 1)) / (mv_floor + 1 - w_yy)
    c2_const = -1.0 / (2 * mh_floor / (wedge_endpoints[-1] - 1) - 1 + 1.0 / (2 * mv_floor / (fwev - 1) - 1))
    c1_const = -c2_const * (2 * mh_floor / (wedge_endpoints[-1] - 1) - 1)
    mask_c = ((w_xx - 1) / mh_floor == (w_yy - 1) / mv_floor)
    w_xx_mod = jnp.where(mask_c, w_xx - 1, w_xx)
    c_r = c1_const + c2_const * (2 - ((w_xx_mod - 1) / mh_floor + (w_yy - 1) / mv_floor)) / \
           ((w_xx_mod - 1) / mh_floor - (w_yy - 1) / mv_floor)
  else:
    raise ValueError(f"Unknown type_wedge: {type_wedge}")

  wl, _ = fdct_wrapping_window_jax(c_l)
  _, wr = fdct_wrapping_window_jax(c_r)
  
  return wl, wr


@functools.partial(
  jax.jit,
  static_argnames=(
    "length_wedge", "subl", "f_c", "f_r", "ww", "mv", "mh",
    "type_wedge"
  )
)
def get_wedge_window_filters_jax(
  length_wedge: int,
  subl: int,
  l_l: jax.Array,
  f_c: int,
  f_r: int,
  ww: int,
  mv: float,
  mh: float,
  wedge_endpoints: jax.Array,
  wedge_midpoints: jax.Array,
  type_wedge: Literal["left", "right", "regular"] = "regular"
) -> tuple[jax.Array, jax.Array]:
  """Computes the window functions for the given wedge in JAX."""
  # Use wrap_data_jax logic to get coordinates without dummy array.
  xj_dummy = jnp.zeros((length_wedge, ww))
  _, w_xx, w_yy = wrap_data_jax(
    length_wedge, ww, l_l, f_c, xj_dummy, f_r, type_wedge=type_wedge, mh=mh
  )
  
  return get_wedge_window_filters_jax_from_coords(
    w_xx, w_yy, subl, l_l, f_c, f_r, ww, mv, mh, wedge_endpoints, wedge_midpoints, type_wedge
  )




def get_wrapped_filtered_data_right(
  quadrant: int,
  nbangles_perquad: int,
  x_hi: np.ndarray,
  f_r: int,
  mv: float,
  mh: float,
  wedge_endpoints: np.ndarray,
  wedge_midpoints: np.ndarray,
) -> np.ndarray:

  fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
  lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))

  rows_wedge = np.arange(1, lcw + 1)
  ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[-1]) / np.floor(4 * mv)
  l_l = np.round(wedge_endpoints[-2] + sl_w * (rows_wedge - 1)).astype(int)
  f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
        ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))

  wdata, w_xx, w_yy = wrap_data(
    lcw, ww, l_l, f_c, x_hi, f_r, type_wedge="right", mh=mh
  )

  slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[-1]) / np.floor(4 * mv)
  c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
         (w_xx - wedge_midpoints[-1] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c2_const = -1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1 + 1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
  c1_const = -c2_const * (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1)
  mask_c = ((w_xx - 1) / np.floor(4 * mh) == (w_yy - 1) / np.floor(4 * mv))
  w_xx[mask_c] -= 1
  c_c = c1_const + c2_const * (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv))) / \
         ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv))
      
  # Build left and right windows.
  wl_l, _ = fdct_wrapping_window(c_l)
  _, wr_r = fdct_wrapping_window(c_c)

  return wdata * wl_l * wr_r


@functools.partial(
  jax.jit,
  static_argnames=("quadrant", "nbangles_perquad", "f_r", "mv", "mh", "wedge_endpoints", "wedge_midpoints")
)
def get_wrapped_filtered_data_right_jax(
  quadrant: int,
  nbangles_perquad: int,
  x_hi: jax.Array,
  f_r: int,
  mv: float,
  mh: float,
  wedge_endpoints: jax.Array,
  wedge_midpoints: jax.Array,
) -> jax.Array:
  """JAX version of get_wrapped_filtered_data_right."""
  fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
  lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))

  rows_wedge = jnp.arange(1, lcw + 1)
  ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[-1]) / np.floor(4 * mv)
  l_l = jnp.round(wedge_endpoints[-2] + sl_w * (rows_wedge - 1)).astype(jnp.int32)
  f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
        ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))

  wdata, w_xx, w_yy = wrap_data_jax(
    lcw, ww, l_l, f_c, x_hi, f_r, type_wedge="right", mh=mh
  )

  slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[-1]) / np.floor(4 * mv)
  c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
         (w_xx - wedge_midpoints[-1] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c2_const = -1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1 + 1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
  c1_const = -c2_const * (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1)
  
  mask_c = ((w_xx - 1) / np.floor(4 * mh) == (w_yy - 1) / np.floor(4 * mv))
  w_xx = jnp.where(mask_c, w_xx - 1, w_xx)
  
  c_c = c1_const + c2_const * (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv))) / \
         ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv))
      
  # Build left and right windows.
  wl_l, _ = fdct_wrapping_window_jax(c_l)
  _, wr_r = fdct_wrapping_window_jax(c_c)

  return wdata * wl_l * wr_r




def get_wrapped_filtered_data_left(
  quadrant: int,
  nbangles_perquad: int,
  x_hi: np.ndarray,
  mv: float,
  mh: float,
  wedge_endpoints: np.ndarray,
  wedge_midpoints: np.ndarray,
) -> np.ndarray:
  fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
  lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
  ww = int(wedge_endpoints[1] + wedge_endpoints[0] - 1)
  # Frequency domain offsets for the periodic wrapping
  # (f_r: row offset, f_c: col offset).
  f_r = int(np.floor(4 * mv) + 2 - np.ceil((lcw + 1) / 2.0) + \
        ((lcw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
  f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
        ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
  rows_wedge = np.arange(1, lcw + 1)
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[0]) / np.floor(4 * mv)
  l_l = np.round(2 - wedge_endpoints[0] + sl_w * (rows_wedge - 1)).astype(int)

  # Wrap the data.
  wdata, w_xx, w_yy = wrap_data(
    lcw, ww, l_l, f_c, x_hi, f_r, type_wedge="left"
    )

  slope_wedge_right = (np.floor(4 * mh) + 1 - wedge_midpoints[0]) / np.floor(4 * mv)
  c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[1] - wedge_endpoints[0]) * \
        (w_xx - wedge_midpoints[0] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c2_const = 1.0 / (1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[0] - 1) - 1) + \
        1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
  c1_const = c2_const / (2 * np.floor(4 * mv) / (fwev - 1) - 1)
  mask_c = ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv) == 2)
  w_xx[mask_c] += 1
  c_c = c1_const + c2_const * ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv)) / \
          (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv)))
  
  # Build left and right windows.
  wl_l, _ = fdct_wrapping_window(c_c)
  _, wr_r = fdct_wrapping_window(c_r)
  
  return wdata * wl_l * wr_r


@functools.partial(
  jax.jit,
  static_argnames=("quadrant", "nbangles_perquad", "mv", "mh", "wedge_endpoints", "wedge_midpoints")
)
def get_wrapped_filtered_data_left_jax(
  quadrant: int,
  nbangles_perquad: int,
  x_hi: jax.Array,
  mv: float,
  mh: float,
  wedge_endpoints: tuple[float, ...],
  wedge_midpoints: tuple[float, ...],
) -> jax.Array:
  """JAX version of get_wrapped_filtered_data_left."""
  fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
  lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
  ww = int(wedge_endpoints[1] + wedge_endpoints[0] - 1)
  # Frequency domain offsets for the periodic wrapping
  # (f_r: row offset, f_c: col offset).
  f_r = int(np.floor(4 * mv) + 2 - np.ceil((lcw + 1) / 2.0) + \
        ((lcw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
  f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
        ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
  rows_wedge = jnp.arange(1, lcw + 1)
  sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[0]) / np.floor(4 * mv)
  l_l = jnp.round(2 - wedge_endpoints[0] + sl_w * (rows_wedge - 1)).astype(jnp.int32)

  # Wrap the data.
  wdata, w_xx, w_yy = wrap_data_jax(
    lcw, ww, l_l, f_c, x_hi, f_r, type_wedge="left"
    )

  slope_wedge_right = (np.floor(4 * mh) + 1 - wedge_midpoints[0]) / np.floor(4 * mv)
  c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[1] - wedge_endpoints[0]) * \
        (w_xx - wedge_midpoints[0] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c2_const = 1.0 / (1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[0] - 1) - 1) + \
        1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
  c1_const = c2_const / (2 * np.floor(4 * mv) / (fwev - 1) - 1)
  
  mask_c = ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv) == 2)
  w_xx = jnp.where(mask_c, w_xx + 1, w_xx)
  
  c_c = c1_const + c2_const * ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv)) / \
          (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv)))
  
  # Build left and right windows.
  wl_l, _ = fdct_wrapping_window_jax(c_c)
  _, wr_r = fdct_wrapping_window_jax(c_r)

  return wdata * wl_l * wr_r