import numpy as np
from scipy import fft
from typing import List, Optional, Tuple, Union, Literal
from dataclasses import dataclass

@dataclass
class CurveletOptions:
  """Options for the Fast Discrete Curvelet Transform.
  
  Attributes:
    is_real: Whether the transform is real-valued.
    m: Number of rows in the image (height).
    n: Number of columns in the image (width).
    nbscales: Number of scales. If None, it is calculated automatically.
    nbangles_coarse: Number of angles at the coarsest level.
    finest: Type of the finest scale ("wavelets" or "curvelets").
    dtype: Data type for the transform (default complex128).
  """
  is_real: bool = False
  m: Optional[int] = None
  n: Optional[int] = None
  nbscales: Optional[int] = None
  nbangles_coarse: int = 16
  finest: Literal["wavelets", "curvelets"] = "curvelets"
  dtype: np.dtype = np.complex128


def fdct_wrapping_window(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Creates the two halves of a C^inf compactly supported window.
  
  Args:
    x: Input coordinate array.
    
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

def _get_lowpass_1d(n_size: int, m_size: float, wavelet_mode: bool = False) -> np.ndarray:
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


def _get_low_high_pass_2d(
  n1: int,
  m1: float,
  n2: int,
  m2: float,
  wavelet_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
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
    _get_lowpass_1d(n1, m1, wavelet_mode),
    _get_lowpass_1d(n2, m2, wavelet_mode)
  )
  highpass = np.sqrt(np.maximum(0, 1 - lowpass**2))
  return lowpass, highpass


def get_nbangles(
  nbscales: int,
  nbangles_coarse: int,
  finest: Literal["wavelets", "curvelets"]
) -> np.ndarray:
  """ Calculates the number of angles at each scale.
  
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


def _get_wedge_ticks(nbangles_perquad: int, m_horiz: float) -> np.ndarray:
  """ Calculates wedge ticks for a quadrant.

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


def _get_wedge_end_mid_points(nbangles_perquad: int, m_horiz: float) -> np.ndarray:
  """ Calculates wedge endpoints and midpoints for a quadrant.

  Args:
    nbangles_perquad: Number of angles per quadrant.
    m_horiz: Horizontal size parameter.

  Returns:
    Wedge endpoints and midpoints arrays.
  """
  wedge_ticks = _get_wedge_ticks(nbangles_perquad, m_horiz)
  wedge_endpoints = wedge_ticks[1::2]
  wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2.0
  return wedge_endpoints, wedge_midpoints


def fdct_wrapping(
  x: np.ndarray,
  is_real: bool = False,
  finest: Literal["wavelets", "curvelets"] = "curvelets",
  nbscales: Optional[int] = None,
  nbangles_coarse: int = 16,
  dtype: np.dtype = np.complex128
) -> List[List[np.ndarray]]:
  """ Fast Discrete Curvelet Transform via wedge wrapping.
  
  Args:
    x: Input image (2D NumPy array).
    is_real: Whether the transform is real-valued.
    finest: Type of the finest scale.
    nbscales: Number of scales.
    nbangles_coarse: Number of angles at the coarsest level.
    
  Returns:
    A list of lists containing curvelet coefficients.
  """
  x = np.asarray(x, dtype=dtype if not is_real else np.real(np.zeros(1, dtype=dtype)).dtype)
  n1, n2 = x.shape
  
  if nbscales is None:
    nbscales = int(np.ceil(np.log2(min(n1, n2)) - 3))
    
  xf = fft.fftshift(fft.fft2(fft.ifftshift(x))) / np.sqrt(x.size)
  nbangles = get_nbangles(nbscales, nbangles_coarse, finest)
  c_coeffs: List[List[Optional[np.ndarray]]] = [[None] * nbangles[j] for j in range(nbscales)]
  
  m1, m2 = n1 / 3.0, n2 / 3.0
  
  if finest == "curvelets":
    # Finest scale is curvelets.
    big_n1 = 2 * int(np.floor(2 * m1)) + 1
    big_n2 = 2 * int(np.floor(2 * m2)) + 1
    idx1 = np.mod(np.floor(n1/2) - np.floor(2 * m1) + np.arange(big_n1), n1).astype(int)
    idx2 = np.mod(np.floor(n2/2) - np.floor(2 * m2) + np.arange(big_n2), n2).astype(int)
    x_low = xf[np.ix_(idx1, idx2)] * np.outer(_get_lowpass_1d(n1, m1), _get_lowpass_1d(n2, m2))
    scales = range(nbscales - 1, 0, -1)
  else:
    # Finest scale is wavelets.
    m1, m2 = m1 / 2.0, m2 / 2.0
    lowpass, hipass = _get_low_high_pass_2d(n1, m1, n2, m2, True)
    idx1 = np.arange(-int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1) + int(np.ceil((n1 + 1) / 2)) - 1
    idx2 = np.arange(-int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1) + int(np.ceil((n2 + 1) / 2)) - 1
    
    x_low = xf[np.ix_(idx1, idx2)] * lowpass
    x_hi = xf.copy()
    x_hi[np.ix_(idx1, idx2)] *= hipass
    c_coeffs[nbscales - 1][0] = fft.fftshift(fft.ifft2(fft.ifftshift(x_hi))) * np.sqrt(x_hi.size)
    if is_real:
      c_coeffs[nbscales - 1][0] = np.real(c_coeffs[nbscales - 1][0])
    scales = range(nbscales - 2, 0, -1)

  for j in scales:
    m1, m2 = m1 / 2.0, m2 / 2.0
    lowpass_next, hipass = _get_low_high_pass_2d(n1, m1, n2, m2, True)
    x_hi = x_low
    idx1 = np.arange(-int(np.floor(2*m1)), int(np.floor(2*m1)) + 1) + int(np.floor(4 * m1))
    idx2 = np.arange(-int(np.floor(2*m2)), int(np.floor(2*m2)) + 1) + int(np.floor(4 * m2))
    x_low_new = x_low[np.ix_(idx1, idx2)]
    x_hi[np.ix_(idx1, idx2)] = x_low_new * hipass
    x_low = x_low_new * lowpass_next
    
    l_idx_total = 0
    nbquadrants = 2 if is_real else 4
    nbangles_perquad = nbangles[j] // 4
    
    for quadrant in range(1, nbquadrants + 1):
      mh, mv = (m2, m1) if quadrant % 2 == 1 else (m1, m2)
      wedge_endpoints, wedge_midpoints = _get_wedge_end_mid_points(
        nbangles_perquad, mh
      )
      
      def process_wrapped_data(data: np.ndarray, q: int, l_idx: int):
        data_rot = np.rot90(data, -(q-1))
        coeffs = fft.fftshift(fft.ifft2(fft.ifftshift(data_rot))) * np.sqrt(data_rot.size)
        if not is_real:
          c_coeffs[j][l_idx] = coeffs.astype(dtype)
        else:
          real_dtype = np.real(np.zeros(1, dtype=dtype)).dtype
          c_coeffs[j][l_idx] = (np.sqrt(2) * np.real(coeffs)).astype(real_dtype)
          c_coeffs[j][l_idx + nbangles[j] // 2] = (np.sqrt(2) * np.imag(coeffs)).astype(real_dtype)

      # 1. Left corner wedge.
      curr_l = l_idx_total;
      l_idx_total += 1
      
      fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
      lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
      ww = int(wedge_endpoints[1] + wedge_endpoints[0] - 1)
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((lcw + 1) / 2.0) + \
            ((lcw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
      y_c = np.arange(1, lcw + 1)
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[0]) / np.floor(4 * mv)
      l_l = np.round(2 - wedge_endpoints[0] + sl_w * (y_c - 1)).astype(int)
      wdata = np.zeros((lcw, ww), dtype=np.complex128)

      w_xx, w_yy = np.zeros_like(wdata, dtype=float), np.zeros_like(wdata, dtype=float)
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
        nr = 1 + np.mod(r - f_r, lcw)
        wdata[nr - 1, :] = x_hi[r - 1, adm - 1] * (cols > 0)
        w_xx[nr - 1, :] = adm; w_yy[nr - 1, :] = r

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
      wl_l, _ = fdct_wrapping_window(c_c); _, wr_r = fdct_wrapping_window(c_r)
      process_wrapped_data(wdata * wl_l * wr_r, quadrant, curr_l)

      # 2. Regular wedges.
      length_wedge = int(np.floor(4 * mv) - np.floor(mv))
      signal_y = np.arange(1, length_wedge + 1)
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      for subl in range(2, nbangles_perquad):
        
        # Current and global indices.
        curr_l = l_idx_total
        l_idx_total += 1
        ww = int(wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1)
        sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[subl - 1]) / np.floor(4 * mv)
        l_l = np.round(wedge_endpoints[subl - 2] + sl_w * (signal_y - 1)).astype(int)
        f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
              ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
        wdata = np.zeros((length_wedge, ww), dtype=np.complex128)
        w_xx, w_yy = np.zeros_like(wdata, dtype=float), np.zeros_like(wdata, dtype=float)
        
        # Aggregating the data.
        for r_idx, r in enumerate(signal_y):
          cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
          nr = 1 + np.mod(r - f_r, length_wedge)
          wdata[nr - 1, :] = x_hi[r - 1, cols - 1]
          w_xx[nr - 1, :] = cols; w_yy[nr - 1, :] = r
        
        # Compute window functions.
        # Compute the slopes of the wedges.
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

        # Apply the window functions and then process the wrapped data.
        process_wrapped_data(wdata * wl_l * wr_r, quadrant, curr_l)

      # 3. Right corner wedge.
      curr_l = l_idx_total; l_idx_total += 1
      ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[-1]) / np.floor(4 * mv)
      l_l = np.round(wedge_endpoints[-2] + sl_w * (y_c - 1)).astype(int)
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
      wdata = np.zeros((lcw, ww), dtype=np.complex128)
      w_xx, w_yy = np.zeros_like(wdata, dtype=float), np.zeros_like(wdata, dtype=float)
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 2 * np.floor(4 * mh) + 1 - np.abs(cols - (2 * np.floor(4 * mh) + 1)))).astype(int)
        nr = 1 + np.mod(r - f_r, lcw)
        wdata[nr - 1, :] = x_hi[r - 1, adm - 1] * (cols <= 2 * np.floor(4 * mh) + 1)
        w_xx[nr - 1, :] = adm; w_yy[nr - 1, :] = r
      slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[-1]) / np.floor(4 * mv)
      c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
             (w_xx - wedge_midpoints[-1] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
      c2_const = -1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1 + 1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
      c1_const = -c2_const * (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1)
      mask_c = ((w_xx - 1) / np.floor(4 * mh) == (w_yy - 1) / np.floor(4 * mv))
      w_xx[mask_c] -= 1
      c_c = c1_const + c2_const * (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv))) / \
             ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv))
      wl_l, _ = fdct_wrapping_window(c_l); _, wr_r = fdct_wrapping_window(c_c)
      process_wrapped_data(wdata * wl_l * wr_r, quadrant, curr_l)
      if quadrant < nbquadrants:
        x_hi = np.rot90(x_hi)

  c_coeffs[0][0] = fft.fftshift(fft.ifft2(fft.ifftshift(x_low))) * np.sqrt(x_low.size)
  if is_real:
    c_coeffs[0][0] = np.real(c_coeffs[0][0])
  
  # Cast to final result type.
  return [[arr for arr in scale] for scale in c_coeffs]

def ifdct_wrapping(
  c_coeffs: List[List[np.ndarray]],
  is_real: bool = False,
  m_img: Optional[int] = None,
  n_img: Optional[int] = None,
  dtype: np.dtype = np.complex128
) -> np.ndarray:
  """
  Inverse Fast Discrete Curvelet Transform via wedge wrapping.
  
  Args:
    c_coeffs: Curvelet coefficients (list of lists of arrays).
    is_real: Whether the transform is real-valued.
    m_img: Target image height.
    n_img: Target image width.
    
  Returns:
    Reconstructed image as a NumPy array.
  """
  nbscales = len(c_coeffs)
  finest: Literal["wavelets", "curvelets"] = "wavelets" if len(c_coeffs[-1]) == 1 else "curvelets"
  nbangles_coarse = len(c_coeffs[1]) if nbscales > 1 else 16
  nbangles = get_nbangles(nbscales, nbangles_coarse, finest)
  n1, n2 = (m_img, n_img) if (m_img and n_img) else c_coeffs[-1][0].shape
  m1, m2 = n1 / 3.0, n2 / 3.0
  
  # Ensure internal xf/xj are complex to handle frequency domain ops
  complex_dtype = np.complex128 if np.issubdtype(dtype, np.floating) else dtype

  if finest == "curvelets":
    xf = np.zeros((2 * int(np.floor(2 * m1)) + 1, 2 * int(np.floor(2 * m2)) + 1), dtype=complex_dtype)
    lowpass = np.outer(_get_lowpass_1d(n1, m1), _get_lowpass_1d(n2, m2))
    scales = range(nbscales - 1, 0, -1)
  else:
    m1, m2 = m1 / 2.0, m2 / 2.0
    xf = np.zeros((2 * int(np.floor(2 * m1)) + 1, 2 * int(np.floor(2 * m2)) + 1), dtype=complex_dtype)
    lowpass = np.outer(_get_lowpass_1d(n1, m1, True), _get_lowpass_1d(n2, m2, True))
    hipass_finest = np.sqrt(np.maximum(0, 1 - lowpass**2))
    scales = range(nbscales - 2, 0, -1)
    
  top_left_1 = top_left_2 = 1
  current_lowpass = lowpass
  
  for j in scales:
    m1, m2 = m1 / 2.0, m2 / 2.0
    lowpass_scale, hipass_scale = _get_low_high_pass_2d(n1, m1, n2, m2, True)
    xj = np.zeros((2 * int(np.floor(4 * m1)) + 1, 2 * int(np.floor(4 * m2)) + 1), dtype=complex_dtype)
    
    nb_per = nbangles[j] // 4
    nbquadrants = 2 if is_real else 4
    l_idx_quad = 0
    for quadrant in range(1, nbquadrants + 1):
      mh, mv = (m2, m1) if quadrant % 2 == 1 else (m1, m2)
      wedge_endpoints, wedge_midpoints = _get_wedge_end_mid_points(nb_per, mh)
      
      def get_wrapped_data(l_idx: int) -> np.ndarray:
        if not is_real:
          x_coeff = c_coeffs[j][l_idx]
        else:
          x_coeff = (c_coeffs[j][l_idx] + 1j * c_coeffs[j][l_idx + nbangles[j] // 2]) / np.sqrt(2.0)
        return np.rot90(fft.fftshift(fft.fft2(fft.ifftshift(x_coeff))) / np.sqrt(x_coeff.size), quadrant - 1)
      
      fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nb_per) + 1))
      lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
      ww = int(wedge_endpoints[1] + wedge_endpoints[0] - 1)
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((lcw + 1) / 2.0) + \
            ((lcw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
      y_c = np.arange(1, lcw + 1)
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[0]) / np.floor(4 * mv)
      l_l = np.round(2 - wedge_endpoints[0] + sl_w * (y_c - 1)).astype(int)
      w_xx, w_yy = np.zeros((lcw, ww)), np.zeros((lcw, ww))
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
        nr = 1 + np.mod(r - f_r, lcw)
        w_xx[nr - 1, :] = adm; w_yy[nr - 1, :] = r
        
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
      wl_l, _ = fdct_wrapping_window(c_c); _, wr_r = fdct_wrapping_window(c_r)
      wdata = get_wrapped_data(l_idx_quad) * wl_l * wr_r
      
      # Aggregating the data into x_j.
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
        nr = 1 + np.mod(r - f_r, lcw)
        xj[r - 1, adm - 1] += wdata[1 + np.mod(r - f_r, lcw) - 1, :]
      l_idx_quad += 1
      
      lw = int(np.floor(4 * mv) - np.floor(mv))
      signal_y = np.arange(1, lw + 1)
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((lw + 1) / 2.0) + \
            ((lw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      for subl in range(2, nb_per):
        ww = int(wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1)
        sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[subl - 1]) / np.floor(4 * mv)
        l_l = np.round(wedge_endpoints[subl - 2] + sl_w * (signal_y - 1)).astype(int)
        f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
              ((ww+1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
        w_xx, w_yy = np.zeros((lw, ww)), np.zeros((lw, ww))
        for r_idx, r in enumerate(signal_y):
          cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
          w_xx[1+np.mod(r-f_r, lw)-1, :] = cols
          w_yy[1+np.mod(r-f_r, lw)-1, :] = r
        slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 2]) / np.floor(4 * mv)
        c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
               (w_xx - wedge_midpoints[subl - 2] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
        slope_wedge_right = (np.floor(4 * mh) + 1 - wedge_midpoints[subl - 1]) / np.floor(4 * mv)
        c_r = 0.5 + np.floor(4 * mv) / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
                (w_xx - wedge_midpoints[subl - 1] - slope_wedge_right * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
        wl_l, _ = fdct_wrapping_window(c_l); _, wr_r = fdct_wrapping_window(c_r)
        wdata = get_wrapped_data(l_idx_quad) * wl_l * wr_r
        
        # Aggregating the data into x_j.
        for r_idx, r in enumerate(signal_y):
          cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
          xj[r - 1, cols - 1] += wdata[1 + np.mod(r - f_r, lw) - 1, :]
        l_idx_quad += 1
        
      ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[-1]) / np.floor(4 * mv)
      l_l = np.round(wedge_endpoints[-2] + sl_w * (y_c - 1)).astype(int)
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww+1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
      w_xx, w_yy = np.zeros((lcw, ww)), np.zeros((lcw, ww))
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 2 * np.floor(4 * mh) + 1 - np.abs(cols - (2 * np.floor(4 * mh) + 1)))).astype(int)
        w_xx[1+np.mod(r-f_r, lcw)-1, :] = adm
        w_yy[1+np.mod(r-f_r, lcw)-1, :] = r
      slope_wedge_left = (np.floor(4 * mh) + 1 - wedge_midpoints[-1]) / np.floor(4 * mv)
      c_l = 0.5 + np.floor(4 * mv) / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
             (w_xx - wedge_midpoints[-1] - slope_wedge_left * (w_yy - 1)) / (np.floor(4 * mv) + 1 - w_yy)
      c2_const = -1.0 / (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1 + 1.0 / (2 * np.floor(4 * mv) / (fwev - 1) - 1))
      c1_const = -c2_const * (2 * np.floor(4 * mh) / (wedge_endpoints[-1] - 1) - 1)
      mask_c = ((w_xx - 1) / np.floor(4 * mh) == (w_yy - 1) / np.floor(4 * mv))
      w_xx[mask_c] -= 1
      c_c = c1_const + c2_const * (2 - ((w_xx - 1) / np.floor(4 * mh) + (w_yy - 1) / np.floor(4 * mv))) / \
             ((w_xx - 1) / np.floor(4 * mh) - (w_yy - 1) / np.floor(4 * mv))
      wl_l, _ = fdct_wrapping_window(c_l); _, wr_c = fdct_wrapping_window(c_c)
      wdata = get_wrapped_data(l_idx_quad) * wl_l * wr_c
      for r_idx, r in enumerate(y_c):
        cols = l_l[r_idx] + np.mod(np.arange(ww) - (l_l[r_idx] - f_c), ww)
        adm = np.round(0.5 * (cols + 2 * np.floor(4 * mh) + 1 - np.abs(cols - (2 * np.floor(4 * mh) + 1)))).astype(int)
        xj[r - 1, adm[::-1] - 1] += wdata[1+np.mod(r-f_r, lcw)-1, ::-1]
      l_idx_quad += 1
      xj = np.rot90(xj)
      
    xj *= current_lowpass
    idx1 = np.arange(-int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1) + int(np.floor(4 * m1))
    idx2 = np.arange(-int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1) + int(np.floor(4 * m2))
    xj[np.ix_(idx1, idx2)] *= hipass_scale
    location_1, location_2 = np.arange(xj.shape[0]) + top_left_1 - 1, np.arange(xj.shape[1]) + top_left_2 - 1
    xf[np.ix_(location_1, location_2)] += xj
    top_left_1 += int(np.floor(4 * m1) - np.floor(2 * m1))
    top_left_2 += int(np.floor(4 * m2) - np.floor(2 * m2))
    current_lowpass = lowpass_scale
    
  if is_real:
    xf = np.rot90(xf, 2) + np.conj(xf)

  # Computing the coarsest scale. 
  xj_coarse = fft.fftshift(fft.fft2(fft.ifftshift(c_coeffs[0][0]))) / np.sqrt(c_coeffs[0][0].size)
  location_1, location_2 = np.arange(xj_coarse.shape[0]) + top_left_1 - 1, np.arange(xj_coarse.shape[1]) + top_left_2 - 1
  xf[np.ix_(location_1, location_2)] += xj_coarse * current_lowpass
  
  m1, m2 = n1 / 3.0, n2 / 3.0
  if finest == "curvelets":
    s1, s2 = int(np.floor(2 * m1) - np.floor(n1 / 2)), int(np.floor(2 * m2) - np.floor(n2 / 2))
    y_folded = xf[:, (np.arange(1, n2 + 1) + s2) - 1]
    y_folded[:, (n2 - s2 + np.arange(1, s2 + 1)) - 1] += xf[:, np.arange(1, s2 + 1) - 1]
    y_folded[:, np.arange(1, s2 + 1) - 1] += xf[:, (n2 + s2 + np.arange(1, s2 + 1)) - 1]
    xf_res = y_folded[(np.arange(1, n1 + 1) + s1) - 1, :]
    xf_res[(n1 - s1 + np.arange(1, s1 + 1)) - 1, :] += y_folded[np.arange(1, s1 + 1) - 1, :]
    xf_res[np.arange(1, s1 + 1) - 1, :] += y_folded[(n1 + s1 + np.arange(1, s1 + 1)) - 1, :]
    xf = xf_res
  else:
    xf_hi_target = fft.fftshift(fft.fft2(fft.ifftshift(c_coeffs[nbscales - 1][0]))) / np.sqrt(c_coeffs[nbscales - 1][0].size)
    lo1, lo2 = np.arange(2 * int(np.floor(m1)) + 1) + int(np.ceil((n1 + 1) / 2.0) - np.floor(m1)) - 1, \
           np.arange(2 * int(np.floor(m2)) + 1) + int(np.ceil((n2 + 1) / 2.0) - np.floor(m2)) - 1
    xf_hi_target[np.ix_(lo1, lo2)] = xf_hi_target[np.ix_(lo1, lo2)] * hipass_finest + xf
    xf = xf_hi_target
    
  x_rec = fft.fftshift(fft.ifft2(fft.ifftshift(xf))) * np.sqrt(xf.size)
  return np.real(x_rec).astype(dtype) if is_real else x_rec.astype(dtype)


def fdct(x: np.ndarray, options: CurveletOptions) -> List[List[np.ndarray]]:
  """Entry point for the forward Fast Discrete Curvelet Transform.
  
  Args:
    x: Input image.
    options: CurveletOptions object containing transform parameters.
    
  Returns:
    Curvelet coefficients as a list of lists of arrays.
  """
  # Ensure options dimensions are set if not provided.
  if options.m is None or options.n is None:
    options.m, options.n = x.shape
    
  # Cast input to specified dtype or appropriate real/complex type.
  if options.is_real:
    x_input = x.astype(
      np.float64 if options.dtype == np.complex128 else options.dtype
    )
  else:
    x_input = x.astype(options.dtype)
    
  return fdct_wrapping(
    x_input,
    is_real=options.is_real,
    finest=options.finest,
    nbscales=options.nbscales,
    nbangles_coarse=options.nbangles_coarse,
    dtype=options.dtype
  )


def ifdct(
  c_coeffs: List[List[np.ndarray]],
  options: CurveletOptions
) -> np.ndarray:
  """Entry point for the inverse Fast Discrete Curvelet Transform.
  
  Args:
    c_coeffs: Curvelet coefficients.
    options: CurveletOptions object containing transform parameters.
    
  Returns:
    Reconstructed image as a NumPy array.
  """
  return ifdct_wrapping(
    c_coeffs,
    is_real=options.is_real,
    m_img=options.m,
    n_img=options.n,
    dtype=options.dtype
  )
