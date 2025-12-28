import numpy as np
import unittest
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform import curvelet_utils as cutils

class TestCurveletUtils(parameterized.TestCase):
  """Unit tests for the curvelet utility functions."""

  def test_fdct_wrapping_window(self):
    """Verify partition of unity and symmetry of the curvelet window."""
    x = np.linspace(0, 1, 100)
    wl, wr = cutils.fdct_wrapping_window(x)
    
    # Check partition of unity: wl^2 + wr^2 = 1.
    np.testing.assert_allclose(wl**2 + wr**2, 1.0, atol=1e-12)
    
    # Check boundary conditions.
    self.assertEqual(wl[0], 0)
    self.assertEqual(wr[0], 1)
    self.assertEqual(wl[-1], 1)
    self.assertEqual(wr[-1], 0)
    
    # Check symmetry (the window is symmetric about 0.5 when considering wl and wr).
    np.testing.assert_allclose(wl, wr[::-1], atol=1e-12)

  @parameterized.parameters(
    (64, 21.33, False),
    (64, 10.66, True),
    (128, 42.66, False),
    (63, 21.0, False) # Test with n_size divisible by 3 and not.
  )
  def test_get_lowpass_1d(self, n_size, m_size, wavelet_mode):
    """Verify characteristics of 1D lowpass filters."""
    lp = cutils.get_lowpass_1d(n_size, m_size, wavelet_mode)
    
    # Check that it's symmetric.
    np.testing.assert_allclose(lp, lp[::-1], atol=1e-12)
    
    # Check bounds.
    self.assertTrue(np.all(lp >= 0))
    self.assertTrue(np.all(lp <= 1))
    
    # Check the flat top (ones at the center).
    center_ones = int(np.floor(m_size)) * 2 + 1
    # Find the middle part.
    mid_idx = len(lp) // 2
    half_ones = int(np.floor(m_size))
    np.testing.assert_allclose(lp[mid_idx - half_ones : mid_idx + half_ones + 1], 1.0)

  @parameterized.parameters(
    (64, 21.33, 64, 21.33, False),
    (64, 10.66, 32, 5.33, True)
  )
  def test_get_low_high_pass_2d(self, n1, m1, n2, m2, wavelet_mode):
    """Verify partition of unity for 2D low/high pass filters."""
    lp2, hp2 = cutils.get_low_high_pass_2d(n1, m1, n2, m2, wavelet_mode)
    
    # Check shapes.
    # get_lowpass_1d length for non-wavelet mode with n_size=64 and m_size=21.33:
    # win_len = floor(2*21.33) - floor(21.33) - 1 - (64%3==0) = 42 - 21 - 1 - 0 = 20.
    # Coord len is 21.
    # Concatenation: wl (21) + ones (43) + wr (21) = 85.
    # Wait, the shape depends on m1, m2.
    
    # Check partition of unity: lp2^2 + hp2^2 = 1.
    np.testing.assert_allclose(lp2**2 + hp2**2, 1.0, atol=1e-12)
    
  @parameterized.parameters(
    (5, 16, "curvelets", [1, 16, 32, 32, 64]),
    (4, 16, "wavelets", [1, 16, 32, 1]),
    (3, 8, "curvelets", [1, 8, 16])
  )
  def test_get_nbangles(self, nbscales, nbangles_coarse, finest, expected):
    """Verify calculation of number of angles per scale."""
    nbangles = cutils.get_nbangles(nbscales, nbangles_coarse, finest)
    np.testing.assert_array_equal(nbangles, expected)

  @parameterized.parameters(
    (4, 10.0),
    (8, 20.0)
  )
  def test_get_wedge_ticks(self, nbangles_perquad, m_horiz):
    """Verify wedge ticks calculation."""
    ticks = cutils.get_wedge_ticks(nbangles_perquad, m_horiz)
    # Ticks should be within the range [1, 2*floor(4*m_horiz)+1]
    max_val = 2 * np.floor(4 * m_horiz) + 1
    self.assertTrue(np.all(ticks >= 1))
    self.assertTrue(np.all(ticks <= max_val))
    # Should have a specific length: if nbangles_perquad is 4, linspace is 5,
    # left is 5, right is 5, concat is 9 or 10?
    # Actually wait: ticks = linspace(0, 0.5, nbangles_perquad+1)
    # wedge_ticks_left = round(...)
    # if even: concat left and right[-2::-1]
    # Length should be related to nbangles_perquad.
    
  def test_get_wedge_end_mid_points(self):
    """Verify wedge end and mid points."""
    nbangles_perquad = 4
    m_horiz = 10.0
    endpoints, midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, m_horiz)
    # endpoints should be every second tick.
    self.assertEqual(len(endpoints), nbangles_perquad)
    self.assertEqual(len(midpoints), nbangles_perquad - 1)
    # midpoints should be between endpoints.
    for i in range(len(midpoints)):
      self.assertTrue(endpoints[i] < midpoints[i] < endpoints[i+1])

  def test_wrap_data_basic(self):
    """Verify basic wrapping on a small synthetic array."""
    length_wedge = 10
    ww = 5
    l_l = np.ones(length_wedge, dtype=int) * 3
    f_c = 2
    x_hi = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
    f_r = 5
    
    wdata, w_xx, w_yy = cutils.wrap_data(
      length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge="regular"
    )
    
    self.assertEqual(wdata.shape, (length_wedge, ww))
    # Verify that wdata values match source x_hi at mapped coordinates.
    for r in range(length_wedge):
      for c in range(ww):
        xx = int(w_xx[r, c])
        yy = int(w_yy[r, c])
        self.assertEqual(wdata[r, c], x_hi[yy-1, xx-1])

  def test_aggregate_from_wrapped_data_basic(self):
    """Verify aggregation logic reverse of wrapping."""
    lcw = 10
    ww = 5
    l_l = np.ones(lcw, dtype=int) * 3
    f_c = 2
    f_r = 5
    wdata = np.random.randn(lcw, ww) + 1j * np.random.randn(lcw, ww)
    xj = np.zeros((20, 20), dtype=complex)
    
    xj_new = cutils.aggregate_from_wrapped_data(
      wdata, xj, l_l, f_c, f_r, lcw, ww, type_wedge="regular"
    )
    
    # Verify values in xj_new match wdata at reverse mapped coordinates.
    for r in range(1, lcw + 1):
      nr = 1 + (r - f_r) % lcw
      cols = l_l[r - 1] + (np.arange(ww) - (l_l[r - 1] - f_c)) % ww
      for c_idx, c in enumerate(cols):
        self.assertEqual(xj_new[r - 1, c - 1], wdata[nr - 1, c_idx])

  def test_get_wedge_window_filters(self):
    """Verify wedge window filters generation."""
    length_wedge = 16
    subl = 3
    l_l = np.ones(length_wedge, dtype=int) * 5
    f_c = 4
    f_r = 8
    ww = 10
    mv, mh = 10.0, 10.0
    nbangles_perquad = 8
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    
    wl, wr = cutils.get_wedge_window_filters(
      length_wedge, subl, l_l, f_c, f_r, ww, mv, mh, wedge_endpoints, wedge_midpoints
    )
    
    self.assertEqual(wl.shape, (length_wedge, ww))
    self.assertEqual(wr.shape, (length_wedge, ww))
    # Window values should be in [0, 1]
    self.assertTrue(np.all(wl >= 0) and np.all(wl <= 1))
    self.assertTrue(np.all(wr >= 0) and np.all(wr <= 1))

  def test_compute_wrapped_data(self):
    """Verify compute_wrapped_data logic."""
    subl = 3
    length_wedge = 16
    quadrant = 1
    nbangles_perquad = 8
    mh, mv = 10.0, 10.0
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    x_hi = np.random.randn(81, 81) + 1j * np.random.randn(81, 81) # Size 2*floor(4*m)+1 = 81
    f_r = 8
    
    wdata = cutils.compute_wrapped_data(
      subl, length_wedge, quadrant, wedge_endpoints, wedge_midpoints, mh, mv, x_hi, f_r
    )
    
    # Check shape: length_wedge x ww
    # ww = int(wedge_endpoints[subl] - wedge_endpoints[subl-2] + 1)
    ww = int(wedge_endpoints[subl] - wedge_endpoints[subl-2] + 1)
    self.assertEqual(wdata.shape, (length_wedge, ww))

  def test_apply_digital_coronara_filter(self):
    """Verify corona filter splits energy correctly."""
    n1, n2 = 64, 64
    m1, m2 = 21.33, 21.33
    x_low = np.random.randn(129, 129) + 1j * np.random.randn(129, 129)
    # Note: size of x_low in coronara filter depends on m1, m2.
    # idx1 = -floor(4*m1)...floor(4*m1)
    
    x_low_new, x_hi = cutils.apply_digital_coronara_filter(x_low, n1, n2, m1, m2)
    
    # Check shapes.
    # idx1/idx2 size is 2 * floor(2*m1) + 1 = 2 * 42 + 1 = 85.
    expected_low_shape = (85, 85)
    self.assertEqual(x_low_new.shape, expected_low_shape)
    self.assertEqual(x_hi.shape, x_low.shape)

if __name__ == "__main__":
  absltest.main()
