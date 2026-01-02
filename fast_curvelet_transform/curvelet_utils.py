""" Utility functions for the Fast Discrete Curvelet Transform (Wrapping)
-------------------------------------------------------------------
This module provides lower-level utility functions for frequency domain 
partitioning, windowing, and data wrapping used by the FDCT.
"""
import numpy as np
from scipy import fft
from typing import List, Tuple, Literal


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
    
  Returns:
    Tuple of (wdata, w_xx, w_yy).
  """
  if type_wedge == "right" and mh is None:
    raise ValueError("mh must be provided for right corner wedge.")
    
  wdata = np.zeros((length_wedge, ww), dtype=np.complex128)
  w_xx, w_yy = np.zeros_like(wdata, dtype=float), np.zeros_like(wdata, dtype=float)
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


def apply_digital_coronara_filter(
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
  
  # Quantities that do not depend on the subl index.
  # TODO: Perhaps we compute them once and pass them as arguments.
  length_wedge = int(np.floor(4 * mv) - np.floor(mv))
  rows_wedge = np.arange(1, length_wedge + 1)
  f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
  

  # Quantities that depend on the subl index.
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
  slope_wedge_left = (
    np.floor(4 * mh) + 1 - wedge_midpoints[subl - 2]
  ) / np.floor(4 * mv)
  slope_wedge_right = (
    np.floor(4 * mh) + 1 - wedge_midpoints[subl - 1]
  ) / np.floor(4 * mv)
  
  # Compute coordinates.
  c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
          (w_xx - wedge_midpoints[subl - 2] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
          (w_xx - wedge_midpoints[subl - 1] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
  
  # Compute window functions.
  wl_l, _ = fdct_wrapping_window(c_l)
  _, wr_r = fdct_wrapping_window(c_r)
  
  return wdata * wl_l * wr_r


def aggregate_from_wrapped_data(
  wdata: np.ndarray,
  xj: np.ndarray,
  l_l: np.ndarray,
  f_c: int,
  f_r: int,
  lcw: int,
  ww: int,
  type_wedge: Literal["left", "right", "regular"] = "regular"
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
  
  Returns:
    Aggregated data array.
  """
  rows_wedge = np.arange(1, lcw + 1)
  
  for r_idx, r in enumerate(rows_wedge):
    cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
    if type_wedge == "left":
      adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
    # elif type_wedge == "right":
    #   adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
    elif type_wedge == "regular":
      adm = cols
    nr = 1 + np.mod(r - f_r, lcw)
    xj[r - 1, adm - 1] += wdata[nr - 1, :]
    
  return xj


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