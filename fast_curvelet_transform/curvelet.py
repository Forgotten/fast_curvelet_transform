""" Fast Discrete Curvelet Transform (Wrapping)
-------------------------------------------
This module provides a Python implementation of the Fast Discrete Curvelet 
Transform (FDCT) using the wrapping approach. 

The transform is implemented as a semi-tight frame. It achieves directional 
selectivity by partitioning the frequency domain into wedges. 'Wrapping' 
is used to efficiently transition between a wedge (in the frequency plane) 
and a rectangular block (suitable for FFT) while preserving the inner product.
"""
import numpy as np
from scipy import fft
from typing import List, Tuple, Literal
from dataclasses import dataclass
import functools
from . import curvelet_utils as cutils


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
  m: int | None = None
  n: int | None = None
  nbscales: int | None = None
  nbangles_coarse: int = 16
  finest: Literal["wavelets", "curvelets"] = "curvelets"
  dtype: np.dtype = np.complex128



def fdct_wrapping(
  x: np.ndarray,
  is_real: bool = False,
  finest: Literal["wavelets", "curvelets"] = "curvelets",
  nbscales: int | None = None,
  nbangles_coarse: int = 16,
  dtype: np.dtype = np.complex128
) -> List[List[np.ndarray]]:
  """Fast Discrete Curvelet Transform via wedge wrapping.
  
  This implementation partitions the frequency domain into concentric squares 
  (scales) and then into angular wedges. For each wedge, the frequency data 
  is 'wrapped' onto a central rectangle and transformed back to the spatial 
  domain via an Inverse FFT.
  
  Args:
    x: Input image (2D NumPy array).
    is_real: Whether the transform is real-valued (returns Hermitian symmetric
      coeffs).
    finest: Type of the finest scale ('wavelets' for isotropic, 'curvelets' for
      directional).
    nbscales: Number of scales.
    nbangles_coarse: Number of angles at the coarsest level.
    dtype: Numpy data type for calculations.
    
  Returns:
    A list of lists containing curvelet coefficients (j, l).
  """
  x = np.asarray(
    x, dtype=dtype if not is_real else np.real(np.zeros(1, dtype=dtype)).dtype
  )
  n1, n2 = x.shape
  
  if nbscales is None:
    nbscales = int(np.ceil(np.log2(min(n1, n2)) - 3))
    
  xf = fft.fftshift(fft.fft2(fft.ifftshift(x))) / np.sqrt(x.size)
  nbangles = cutils.get_nbangles(nbscales, nbangles_coarse, finest)
  c_coeffs: List[List[np.ndarray | None]] = [
    [None] * nbangles[j] for j in range(nbscales)
  ]
  
  m1, m2 = n1 / 3.0, n2 / 3.0
  
  if finest == "curvelets":
    # Finest scale is curvelets.
    big_n1 = 2 * int(np.floor(2 * m1)) + 1
    big_n2 = 2 * int(np.floor(2 * m2)) + 1
    idx1 = np.mod(
      np.floor(n1/2) - np.floor(2 * m1) + np.arange(big_n1), n1
    ).astype(int)
    idx2 = np.mod(
      np.floor(n2/2) - np.floor(2 * m2) + np.arange(big_n2), n2
    ).astype(int)
    lowpass, _ = cutils.get_low_high_pass_2d(n1, m1, n2, m2)
    x_low = xf[np.ix_(idx1, idx2)] * lowpass
    scales = range(nbscales - 1, 0, -1)
  else:
    # Finest scale is wavelets.
    m1, m2 = m1 / 2.0, m2 / 2.0
    lowpass, hipass = cutils.get_low_high_pass_2d(n1, m1, n2, m2, True)
    idx1 = np.arange(
      -int(np.floor(2 * m1)), int(np.floor(2 * m1)) + 1
    ) + int(np.ceil((n1 + 1) / 2)) - 1
    idx2 = np.arange(
      -int(np.floor(2 * m2)), int(np.floor(2 * m2)) + 1
    ) + int(np.ceil((n2 + 1) / 2)) - 1
    
    x_low = xf[np.ix_(idx1, idx2)] * lowpass
    x_hi = xf.copy()
    x_hi[np.ix_(idx1, idx2)] *= hipass
    c_coeffs[nbscales - 1][0] = fft.fftshift(
      fft.ifft2(fft.ifftshift(x_hi))
    ) * np.sqrt(x_hi.size)
    if is_real:
      c_coeffs[nbscales - 1][0] = np.real(c_coeffs[nbscales - 1][0])
    scales = range(nbscales - 2, 0, -1)

  for j in scales:
    m1, m2 = m1 / 2.0, m2 / 2.0
    # Localize the information around the Cartesian corona in frequency domain.
    x_low, x_hi = cutils.apply_digital_coronara_filter(x_low, n1, n2, m1, m2)
    
    l_idx = 0 # Global angle index for the current scale.
    nbquadrants = 2 if is_real else 4 # Only 2 quadrants needed for real signals (symmetry).
    nbangles_perquad = nbangles[j] // 4
    
    # Process the frequency plane in quadrants (East, North, West, South).
    # Here depending on the quadrant we rotate the data by 90 degrees, so we can
    # apply the same transform to all quadrants.
    for quadrant in range(1, nbquadrants + 1):
      mh, mv = (m2, m1) if quadrant % 2 == 1 else (m1, m2)
      wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(
        nbangles_perquad, mh
      )
      
      def compute_coeffs_from_wrapped_data(
        wrapped_data: np.ndarray,
        quadrant: int,
        angle_idx: int
      ) -> None:
        """Process wrapped data for a given quadrant.
        
        Args:
          wrapped_data: Input data to process.
          quadrant: Quadrant index.
          angle_idx: Angle index.

        Returns:
          None, but it updates the c_coeffs list in place.
        """
        data_rot = np.rot90(wrapped_data, -(quadrant-1))
        coeffs = fft.fftshift(fft.ifft2(fft.ifftshift(data_rot))) * np.sqrt(data_rot.size)
        if not is_real:
          c_coeffs[j][angle_idx] = coeffs.astype(dtype)
        else:
          real_dtype = np.real(np.zeros(1, dtype=dtype)).dtype
          c_coeffs[j][angle_idx] = (np.sqrt(2) * np.real(coeffs)).astype(real_dtype)
          c_coeffs[j][angle_idx + nbangles[j] // 2] = (np.sqrt(2) * np.imag(coeffs)).astype(real_dtype)

      # 1. Left corner wedge.
      fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
      lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
      ww = int(wedge_endpoints[1] + wedge_endpoints[0] - 1)
      # Frequency domain offsets for the periodic wrapping
      # (f_r: row offset, f_c: col offset).
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((lcw + 1) / 2.0) + \
            ((lcw + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
      y_c = np.arange(1, lcw + 1)
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[0]) / np.floor(4 * mv)
      l_l = np.round(2 - wedge_endpoints[0] + sl_w * (y_c - 1)).astype(int)

      # Wrap the data.
      wdata, w_xx, w_yy = cutils.wrap_data(
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
      wl_l, _ = cutils.fdct_wrapping_window(c_c)
      _, wr_r = cutils.fdct_wrapping_window(c_r)
      compute_coeffs_from_wrapped_data(wdata * wl_l * wr_r, quadrant, l_idx)

      # Increment the angle index.
      l_idx += 1

      # 2. Regular wedges (between the corners).
      length_wedge = int(np.floor(4 * mv) - np.floor(mv))
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      
      # For the regular wedges, we can use partial application to avoid
      # passing the same arguments multiple times.
      _compute_wrapped_data_partial = functools.partial(
        cutils.compute_wrapped_data,
        length_wedge=length_wedge,
        quadrant=quadrant,
        wedge_endpoints=wedge_endpoints,
        wedge_midpoints=wedge_midpoints,
        mh=mh,
        mv=mv,
        x_hi=x_hi,
        f_r=f_r
      )

      # Iterate through the directional wedges within the current quadrant.
      for subl in range(2, nbangles_perquad):
        
        # Compute wrapped data.
        wdata = _compute_wrapped_data_partial(subl)
        
        # Apply the window functions and then process the wrapped data.
        compute_coeffs_from_wrapped_data(wdata, quadrant, l_idx)

        # Increment the angle index.
        l_idx += 1

      # 3. Right corner wedge.
      ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
      sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[-1]) / np.floor(4 * mv)
      l_l = np.round(wedge_endpoints[-2] + sl_w * (y_c - 1)).astype(int)
      f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
            ((ww + 1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))

      wdata, w_xx, w_yy = cutils.wrap_data(
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
      wl_l, _ = cutils.fdct_wrapping_window(c_l)
      _, wr_r = cutils.fdct_wrapping_window(c_c)

      # Apply the window functions and then process the wrapped data.
      compute_coeffs_from_wrapped_data(wdata * wl_l * wr_r, quadrant, l_idx)

      # Rotate the image for the next quadrant.
      if quadrant < nbquadrants:
        x_hi = np.rot90(x_hi)

      # Increment the angle index.
      l_idx += 1

  # Compute the low-pass coefficients.
  c_coeffs[0][0] = fft.fftshift(fft.ifft2(fft.ifftshift(x_low))) * np.sqrt(x_low.size)
  if is_real:
    c_coeffs[0][0] = np.real(c_coeffs[0][0])
  
  # Cast to final result type.
  return [[arr for arr in scale] for scale in c_coeffs]


def ifdct_wrapping(
  c_coeffs: List[List[np.ndarray]],
  is_real: bool = False,
  m_img: int | None = None,
  n_img: int | None = None,
  dtype: np.dtype = np.complex128
) -> np.ndarray:
  """Inverse Fast Discrete Curvelet Transform via wedge wrapping.
  
  Reconstructs the spatial image by wrapping the coefficient spectra back 
  into their respective frequency domain positions (wedges), summing them, 
  and applying a final 2D Inverse FFT.
  
  Args:
    c_coeffs: Curvelet coefficients (list of lists of arrays).
    is_real: Whether the transform is real-valued.
    m_img: Target image height (rows).
    n_img: Target image width (columns).
    dtype: Numpy data type for calculations.
    
  Returns:
    Reconstructed image as a NumPy array.
  """
  nbscales = len(c_coeffs)
  # We determine whether the finest scale is a wavelet or a curvelet.
  finest: Literal["wavelets", "curvelets"] = "wavelets" if len(c_coeffs[-1]) == 1 else "curvelets"
  # Determine the number of coarse angles.
  nbangles_coarse = len(c_coeffs[1]) if nbscales > 1 else 16
  nbangles = cutils.get_nbangles(nbscales, nbangles_coarse, finest)
  n1, n2 = (m_img, n_img) if (m_img and n_img) else c_coeffs[-1][0].shape
  m1, m2 = n1 / 3.0, n2 / 3.0
  
  # Ensure internal xf/xj are complex to handle frequency domain ops
  complex_dtype = np.complex128 if np.issubdtype(dtype, np.floating) else dtype

  if finest == "curvelets":
    xf = np.zeros(
      (2 * int(np.floor(2 * m1)) + 1, 2 * int(np.floor(2 * m2)) + 1),
      dtype=complex_dtype
    )
    lowpass, _ = cutils.get_low_high_pass_2d(n1, m1, n2, m2)
    scales = range(nbscales - 1, 0, -1)
  elif finest == "wavelets":
    m1, m2 = m1 / 2.0, m2 / 2.0
    xf = np.zeros(
      (2 * int(np.floor(2 * m1)) + 1, 2 * int(np.floor(2 * m2)) + 1),
      dtype=complex_dtype
    )
    lowpass, hipass_finest = cutils.get_low_high_pass_2d(n1, m1, n2, m2, True)
    scales = range(nbscales - 2, 0, -1)
  else:
    raise ValueError("Finest scale must be either 'wavelets' or 'curvelets'.")

  top_left_1 = top_left_2 = 1
  current_lowpass = lowpass
  
  for j in scales:
    m1, m2 = m1 / 2.0, m2 / 2.0
    lowpass_scale, hipass_scale = cutils.get_low_high_pass_2d(n1, m1, n2, m2, True)
    xj = np.zeros(
      (2 * int(np.floor(4 * m1)) + 1, 2 * int(np.floor(4 * m2)) + 1),
      dtype=complex_dtype
    )
    
    nb_per = nbangles[j] // 4
    nbquadrants = 2 if is_real else 4
    l_idx_quad = 0
    for quadrant in range(1, nbquadrants + 1):
      mh, mv = (m2, m1) if quadrant % 2 == 1 else (m1, m2)
      wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nb_per, mh)
      
      def _get_wrapped_data(l_idx: int) -> np.ndarray:
        """Extract the wrapped data from the coefficients.
        
        Args:
          l_idx: The index of the coefficient.
        
        Returns:
          The wrapped data after FFT and rotation depending on the quadrant.
        """
        if not is_real:
          x_coeff = c_coeffs[j][l_idx]
        else:
          x_coeff = (
            c_coeffs[j][l_idx] + 1j * c_coeffs[j][l_idx + nbangles[j] // 2]
          ) / np.sqrt(2.0)
        return np.rot90(
          fft.fftshift(
            fft.fft2(
              fft.ifftshift(x_coeff)
            )
          ) / np.sqrt(x_coeff.size), quadrant - 1)
      
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
        w_xx[nr - 1, :] = adm
        w_yy[nr - 1, :] = r
      
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
      
      # Compute the wrapping windows.
      wl_l, _ = cutils.fdct_wrapping_window(c_c)
      _, wr_r = cutils.fdct_wrapping_window(c_r)
      
      # Compute the wrapped data.
      wdata = _get_wrapped_data(l_idx_quad) * wl_l * wr_r
      
      # Aggregating the wrapped data into x_j.
      xj = cutils.aggregate_from_wrapped_data(
        wdata, xj, l_l, f_c, f_r, lcw, ww, type_wedge="left"
      )
      
      l_idx_quad += 1
      
      length_wedge = int(np.floor(4 * mv) - np.floor(mv))
      rows_wedge = np.arange(1, length_wedge + 1)
      f_r = int(np.floor(4 * mv) + 2 - np.ceil((length_wedge + 1) / 2.0) + \
            ((length_wedge + 1) % 2) * (1 if (quadrant - 2) == (quadrant - 2) % 2 else 0))
      
      get_wedge_window_filters_partial = functools.partial(
        cutils.get_wedge_window_filters,
        length_wedge=length_wedge,
        mv=mv,
        mh=mh,
        wedge_endpoints=wedge_endpoints,
        wedge_midpoints=wedge_midpoints)

      for subl in range(2, nb_per):
        
        ww = int(wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1)
        sl_w = (np.floor(4 * mh) + 1 - wedge_endpoints[subl - 1]) / np.floor(4 * mv)
        l_l = np.round(wedge_endpoints[subl - 2] + sl_w * (rows_wedge - 1)).astype(int)
        f_c = int(np.floor(4 * mh) + 2 - np.ceil((ww + 1) / 2.0) + \
              ((ww+1) % 2) * (1 if (quadrant - 3) == (quadrant - 3) % 2 else 0))
        
        # Compute the wrapping windows.
        wl_l, wr_r = get_wedge_window_filters_partial(
          subl=subl, l_l=l_l, f_c=f_c, f_r=f_r, ww=ww)

        # Extract the wrapped data from the coefficients.
        wdata = _get_wrapped_data(l_idx_quad) * wl_l * wr_r
        
        # Unwrapping the data into x_j.
        xj = cutils.aggregate_from_wrapped_data(
          wdata, xj, l_l, f_c, f_r, length_wedge, ww, type_wedge="regular"
        )

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
      wl_l, _ = cutils.fdct_wrapping_window(c_l)
      _, wr_c = cutils.fdct_wrapping_window(c_c)
      wdata = _get_wrapped_data(l_idx_quad) * wl_l * wr_c
      
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
    # For real signals, we only computed the first two quadrants.
    # The rest of the frequency domain is the Hermitian conjugate
    # (flipped and conjugated).
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
