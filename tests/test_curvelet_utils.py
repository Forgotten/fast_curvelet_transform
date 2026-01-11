import numpy as np
import unittest
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform import curvelet_utils as cutils
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

class TestCurveletUtils(parameterized.TestCase):
  """Unit tests for the curvelet utility functions."""

  def test_fdct_wrapping_window(self):
    """Verify partition of unity and symmetry of the curvelet window."""
    x = np.linspace(0, 1, 100)
    wl, wr = cutils.fdct_wrapping_window(x)
    
    # Check partition of unity: wl^2 + wr^2 = 1.
    np.testing.assert_allclose(wl**2 + wr**2, 1.0, atol=1e-7)
    
    # Check boundary conditions.
    self.assertEqual(wl[0], 0)
    self.assertEqual(wr[0], 1)
    self.assertEqual(wl[-1], 1)
    self.assertEqual(wr[-1], 0)
    
    # Check symmetry (the window is symmetric about 0.5 when considering wl and wr).
    np.testing.assert_allclose(wl, wr[::-1], atol=1e-7)

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
    np.testing.assert_allclose(lp, lp[::-1], atol=1e-7)
    
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
    np.testing.assert_allclose(lp2**2 + hp2**2, 1.0, atol=1e-7)
    
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
    (5, 16, "curvelets"),
    (4, 16, "wavelets"),
    (3, 8, "curvelets")
  )
  def test_get_nbangles_jax(self, nbscales, nbangles_coarse, finest):
    """Verify JAX get_nbangles matches NumPy version."""
    expected = cutils.get_nbangles(nbscales, nbangles_coarse, finest)
    nbangles_jax = cutils.get_nbangles_jax(nbscales, nbangles_coarse, finest)
    np.testing.assert_array_equal(nbangles_jax, expected)

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
    quadrant = 1
    nbangles_perquad = 8
    mh, mv = 10.0, 10.0
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    x_hi = np.random.randn(81, 81) + 1j * np.random.randn(81, 81) # Size 2*floor(4*m)+1 = 81
    
    wdata = cutils.compute_wrapped_data(
      subl, quadrant, wedge_endpoints, wedge_midpoints, mh, mv, x_hi
    )
    length_wedge = int(np.floor(4 * mv) - np.floor(mv))
    
    # Check shape: length_wedge x ww
    # ww = int(wedge_endpoints[subl] - wedge_endpoints[subl-2] + 1)
    ww = int(wedge_endpoints[subl] - wedge_endpoints[subl-2] + 1)
    self.assertEqual(wdata.shape, (length_wedge, ww))

  def test_apply_digital_corona_filter(self):
    """Verify corona filter splits energy correctly."""
    n1, n2 = 64, 64
    m1, m2 = 21.33, 21.33
    x_low = np.random.randn(129, 129) + 1j * np.random.randn(129, 129)
    # Note: size of x_low in coronara filter depends on m1, m2.
    # idx1 = -floor(4*m1)...floor(4*m1)
    
    x_low_new, x_hi = cutils.apply_digital_corona_filter(x_low, n1, n2, m1, m2)
    
    # Check shapes.
    # idx1/idx2 size is 2 * floor(2*m1) + 1 = 2 * 42 + 1 = 85.
    expected_low_shape = (85, 85)
    self.assertEqual(x_low_new.shape, expected_low_shape)
    self.assertEqual(x_hi.shape, x_low.shape)

  def test_apply_digital_corona_filter_jax(self):
    """Verify JAX corona filter matches NumPy version."""
    n1, n2 = 64, 64
    m1, m2 = 21.33, 21.33
    x_low = np.random.randn(129, 129) + 1j * np.random.randn(129, 129)
    
    # NumPy
    x_low_new_np, x_hi_np = cutils.apply_digital_corona_filter(x_low, n1, n2, m1, m2)
    
    # JAX
    x_low_new_jax, x_hi_jax = cutils.apply_digital_corona_filter_jax(
      jnp.array(x_low), n1, n2, m1, m2
    )
    
    np.testing.assert_allclose(x_low_new_np, x_low_new_jax, atol=1e-7)
    np.testing.assert_allclose(x_hi_np, x_hi_jax, atol=1e-7)

  def test_curvelet_finest_level(self):
    """Verify curvelet_finest_level utility."""
    n1, n2 = 64, 64
    m1, m2 = n1 / 3.0, n2 / 3.0
    xf = np.random.randn(n1, n2) + 1j * np.random.randn(n1, n2)
    x_low = cutils.curvelet_finest_level(n1, m1, n2, m2, xf)
    # Check shape: 2 * floor(2*m1) + 1 = 2 * 42 + 1 = 85.
    # Wait, n1=64, m1=64/3 = 21.33. floor(2*m1) = 42. So 85x85.
    self.assertEqual(x_low.shape, (85, 85))

  def test_wavelet_finest_level(self):
    """Verify wavelet_finest_level utility."""
    n1, n2 = 64, 64
    m1, m2 = n1 / 6.0, n2 / 6.0 # Wavelet mode often halves m.
    xf = np.random.randn(n1, n2) + 1j * np.random.randn(n1, n2)
    x_low, x_hi = cutils.wavelet_finest_level(n1, m1, n2, m2, xf)
    # x_low size: 2 * floor(2*m1) + 1 = 2 * floor(21.33) + 1 = 43.
    self.assertEqual(x_low.shape, (43, 43))
    self.assertEqual(x_hi.shape, xf.shape)

  def test_fdct_wrapping_window_jax(self):
    """Verify partition of unity for JAX window."""
    x = jnp.linspace(0, 1, 100)
    wl, wr = cutils.fdct_wrapping_window_jax(x)
    np.testing.assert_allclose(wl**2 + wr**2, 1.0, atol=1e-7)
    np.testing.assert_allclose(wl, wr[::-1], atol=1e-7)

  @parameterized.parameters(
    (64, 21.33, False),
    (64, 10.66, True)
  )
  def test_get_lowpass_1d_jax(self, n_size, m_size, wavelet_mode):
    """Verify JAX 1D lowpass filters matches numpy version."""
    lp_np = cutils.get_lowpass_1d(n_size, m_size, wavelet_mode)
    lp_jax = cutils.get_lowpass_1d_jax(n_size, m_size, wavelet_mode)
    np.testing.assert_allclose(lp_np, lp_jax, atol=1e-7)

  @parameterized.parameters(
    (64, 21.33, 64, 21.33, False),
    (64, 10.66, 32, 5.33, True)
  )
  def test_get_low_high_pass_2d_jax(self, n1, m1, n2, m2, wavelet_mode):
    """Verify JAX 2D low/high pass filters match numpy versions."""
    lp_np, hp_np = cutils.get_low_high_pass_2d(n1, m1, n2, m2, wavelet_mode)
    lp_jax, hp_jax = cutils.get_low_high_pass_2d_jax(n1, m1, n2, m2, wavelet_mode)
    np.testing.assert_allclose(lp_np, lp_jax, atol=1e-7)
    np.testing.assert_allclose(hp_np, hp_jax, atol=1e-7)

  def test_get_wedge_window_filters_jax(self):
    """Verify JAX wedge window filters match numpy version."""
    length_wedge = 16
    subl = 3
    l_l = jnp.ones(length_wedge, dtype=int) * 5
    f_c = 4
    f_r = 8
    ww = 10
    mv, mh = 10.0, 10.0
    nbangles_perquad = 8
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    
    wl_np, wr_np = cutils.get_wedge_window_filters(
      length_wedge, subl, np.array(l_l), f_c, f_r, ww, mv, mh, wedge_endpoints, wedge_midpoints
    )
    wl_jax, wr_jax = cutils.get_wedge_window_filters_jax(
      length_wedge, subl, l_l, f_c, f_r, ww, mv, mh, tuple(wedge_endpoints), tuple(wedge_midpoints)
    )
    
    np.testing.assert_allclose(wl_np, wl_jax, atol=1e-7)
    np.testing.assert_allclose(wr_np, wr_jax, atol=1e-7)

  @parameterized.parameters(
    ("regular", None),
    ("left", None),
    ("right", 10)
  )
  def test_wrap_data_jax(self, type_wedge, mh):
    """Verify wrap_data_jax matches wrap_data."""
    length_wedge = 10
    ww = 5
    l_l = np.random.randint(5, 15, size=length_wedge)
    f_c = 2
    x_hi = np.random.randn(20, 60) + 1j * np.random.randn(20, 60)
    f_r = 5
    
    # NumPy version
    wdata_np, w_xx_np, w_yy_np = cutils.wrap_data(
      length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge=type_wedge, mh=mh
    )
    
    # JAX version
    wdata_jax, w_xx_jax, w_yy_jax = cutils.wrap_data_jax(
      length_wedge, ww, jnp.array(l_l), f_c, jnp.array(x_hi), f_r, 
      type_wedge=type_wedge, mh=mh
    )
    
    np.testing.assert_allclose(wdata_np, wdata_jax, atol=1e-7)
    np.testing.assert_allclose(w_xx_np, w_xx_jax, atol=1e-7)
    np.testing.assert_allclose(w_yy_np, w_yy_jax, atol=1e-7)

  @parameterized.parameters(
    ("regular", None),
    ("left", None),
    ("right", 10)
  )
  def test_wrap_data_jax_jit(self, type_wedge, mh):
    """Verify wrap_data_jax can be JIT-compiled."""
    length_wedge = 10
    ww = 5
    l_l = jnp.array(np.random.randint(5, 15, size=length_wedge))
    f_c = 2
    x_hi = jnp.array(np.random.randn(20, 60) + 1j * np.random.randn(20, 60))
    f_r = 5
    
    # Define jitted version. 
    # length_wedge and ww must also be static since they determine shapes.
    # However, following user request to use the last two as static.
    # In this test scope, they are passed as literals to the function.
    jitted_fn = jax.jit(
      cutils.wrap_data_jax,
      static_argnames=("length_wedge", "ww", "f_c", "f_r", "type_wedge", "mh")
    )
    
    # Warmup
    out_jit = jitted_fn(
      length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge=type_wedge, mh=mh
    )
    
    # Original
    out_orig = cutils.wrap_data_jax(
      length_wedge, ww, l_l, f_c, x_hi, f_r, type_wedge=type_wedge, mh=mh
    )
    
    for o_jit, o_orig in zip(out_jit, out_orig):
      np.testing.assert_allclose(o_jit, o_orig, atol=1e-7)

  @parameterized.parameters(
    (np.float32, jnp.float32),
    (np.float64, jnp.float64),
    (np.int32, jnp.int32)
  )
  def test_wrap_data_dtypes(self, np_dtype, jax_dtype):
    """Verify dtype_coord argument works for both wrap_data versions."""
    length_wedge = 10
    ww = 5
    l_l = np.random.randint(5, 15, size=length_wedge)
    f_c = 2
    x_hi = np.random.randn(20, 60) + 1j * np.random.randn(20, 60)
    f_r = 5
    
    # Check NumPy version
    _, w_xx_np, w_yy_np = cutils.wrap_data(
      length_wedge, ww, l_l, f_c, x_hi, f_r, dtype_coord=np_dtype
    )
    self.assertEqual(w_xx_np.dtype, np_dtype)
    self.assertEqual(w_yy_np.dtype, np_dtype)
    
    # Check JAX version
    _, w_xx_jax, w_yy_jax = cutils.wrap_data_jax(
      length_wedge, ww, jnp.array(l_l), f_c, jnp.array(x_hi), f_r, dtype_coord=jax_dtype
    )
    self.assertEqual(w_xx_jax.dtype, jax_dtype)
    self.assertEqual(w_yy_jax.dtype, jax_dtype)

  @parameterized.parameters(
    ("regular", None),
    ("left", None),
    ("right", 10.0),
  )
  def test_aggregate_from_wrapped_data_jax(self, type_wedge, mh):
    """Verify aggregate_from_wrapped_data_jax matches NumPy version."""
    lcw, ww = 10, 5
    l_l = np.random.randint(5, 15, size=lcw)
    f_c, f_r = 2, 5
    wdata = np.random.randn(lcw, ww) + 1j * np.random.randn(lcw, ww)
    xj_np = np.zeros((20, 40), dtype=complex)
    xj_jax = jnp.zeros((20, 40), dtype=complex)
    
    # NumPy
    xj_np = cutils.aggregate_from_wrapped_data(
      wdata, xj_np, l_l, f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    # JAX
    xj_jax = cutils.aggregate_from_wrapped_data_jax(
      jnp.array(wdata), xj_jax, jnp.array(l_l), f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    np.testing.assert_allclose(xj_np, xj_jax, atol=1e-7)

  def test_compute_wrapped_data_jax(self):
    """Verify compute_wrapped_data_jax matches NumPy version."""
    nbangles_perquad = 8
    mh, mv = 10.0, 10.0
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    subl = 3
    quadrant = 1
    x_hi = np.random.randn(81, 81) + 1j * np.random.randn(81, 81)
    
    # NumPy
    wdata_np = cutils.compute_wrapped_data(
      subl, quadrant, wedge_endpoints, wedge_midpoints, mh, mv, x_hi
    )
    
    # JAX
    wdata_jax = cutils.compute_wrapped_data_jax(
      subl, quadrant, tuple(wedge_endpoints), tuple(wedge_midpoints), mh, mv, jnp.array(x_hi)
    )
    
    np.testing.assert_allclose(wdata_np, wdata_jax, atol=1e-7)

  @parameterized.parameters(
    ("regular", None),
    ("left", None),
    ("right", 10.0),
  )
  def test_aggregate_from_wrapped_data_jax_jit(self, type_wedge, mh):
    """Verify aggregate_from_wrapped_data_jax can be JIT-compiled."""
    lcw, ww = 10, 5
    l_l = jnp.array(np.random.randint(5, 15, size=lcw))
    f_c, f_r = 2, 5
    wdata = jnp.array(np.random.randn(lcw, ww) + 1j * np.random.randn(lcw, ww))
    xj = jnp.zeros((20, 40), dtype=complex)
    
    jitted_fn = jax.jit(
      cutils.aggregate_from_wrapped_data_jax,
      static_argnames=("f_c", "f_r", "lcw", "ww", "type_wedge", "mh")
    )
    
    # Warmup
    xj_jit = jitted_fn(wdata, xj, l_l, f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh)
    
    # Original
    xj_orig = cutils.aggregate_from_wrapped_data_jax(
      wdata, xj, l_l, f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    np.testing.assert_allclose(xj_jit, xj_orig, atol=1e-7)

  def test_compute_wrapped_data_jax_jit(self):
    """Verify compute_wrapped_data_jax can be JIT-compiled."""
    nbangles_perquad = 8
    mh, mv = 10.0, 10.0
    wedge_endpoints = tuple(cutils.get_wedge_end_mid_points(nbangles_perquad, mh)[0])
    wedge_midpoints = tuple(cutils.get_wedge_end_mid_points(nbangles_perquad, mh)[1])
    subl = 3
    quadrant = 1
    x_hi = jnp.array(np.random.randn(81, 81) + 1j * np.random.randn(81, 81))
    
    jitted_fn = jax.jit(
      cutils.compute_wrapped_data_jax,
      static_argnames=("subl", "quadrant", "mh", "mv", "wedge_endpoints", "wedge_midpoints")
    )
    
    # Warmup
    wdata_jit = jitted_fn(subl, quadrant, wedge_endpoints, wedge_midpoints, mh, mv, x_hi)
    
    # Original
    wdata_orig = cutils.compute_wrapped_data_jax(
      subl, quadrant, wedge_endpoints, wedge_midpoints, mh, mv, x_hi
    )
    
    np.testing.assert_allclose(wdata_jit, wdata_orig, atol=1e-7)

  def test_curvelet_finest_level_jax(self):
    """Verify curvelet_finest_level_jax matches NumPy version."""
    n1, n2 = 64, 64
    m1, m2 = n1 / 3.0, n2 / 3.0
    xf = np.random.randn(n1, n2) + 1j * np.random.randn(n1, n2)
    
    # NumPy
    x_low_np = cutils.curvelet_finest_level(n1, m1, n2, m2, xf)
    
    # JAX
    x_low_jax = cutils.curvelet_finest_level_jax(n1, m1, n2, m2, jnp.array(xf))
    
    np.testing.assert_allclose(x_low_np, x_low_jax, atol=1e-7)

  def test_wavelet_finest_level_jax(self):
    """Verify wavelet_finest_level_jax matches NumPy version."""
    n1, n2 = 64, 64
    m1, m2 = n1 / 6.0, n2 / 6.0
    xf = np.random.randn(n1, n2) + 1j * np.random.randn(n1, n2)
    
    # NumPy
    x_low_np, x_hi_np = cutils.wavelet_finest_level(n1, m1, n2, m2, xf)
    
    # JAX
    x_low_jax, x_hi_jax = cutils.wavelet_finest_level_jax(n1, m1, n2, m2, jnp.array(xf))
    
    np.testing.assert_allclose(x_low_np, x_low_jax, atol=1e-7)
    np.testing.assert_allclose(x_hi_np, x_hi_jax, atol=1e-7)

  def test_get_wrapped_filtered_data_right_jax(self):
    """Verify get_wrapped_filtered_data_right_jax matches NumPy version."""
    nbangles_perquad = 8
    mv, mh = 10.0, 10.0
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    fwev = int(np.round(2 * np.floor(4 * mv) / (2 * nbangles_perquad) + 1))
    lcw = int(np.floor(4 * mv) - np.floor(mv) + np.ceil(fwev / 4.0))
    ww = int(4 * np.floor(4 * mh) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2])
    f_r = 5
    quadrant = 3
    x_hi = np.random.randn(81, 81) + 1j * np.random.randn(81, 81)
    
    # NumPy
    wdt_np = cutils.get_wrapped_filtered_data_right(
      quadrant, nbangles_perquad, x_hi, f_r, mv, mh, wedge_endpoints, wedge_midpoints
    )
    
    # JAX
    wdt_jax = cutils.get_wrapped_filtered_data_right_jax(
      quadrant, nbangles_perquad, jnp.array(x_hi), f_r, mv, mh, 
      tuple(wedge_endpoints), tuple(wedge_midpoints)
    )
    
    np.testing.assert_allclose(wdt_np, wdt_jax, atol=1e-7)

  def test_get_wrapped_filtered_data_left_jax(self):
    """Verify get_wrapped_filtered_data_left_jax matches NumPy version."""
    nbangles_perquad = 8
    mv, mh = 10.0, 10.0
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    quadrant = 2
    x_hi = np.random.randn(81, 81) + 1j * np.random.randn(81, 81)
    
    # NumPy
    wdt_np = cutils.get_wrapped_filtered_data_left(
      quadrant, nbangles_perquad, x_hi, mv, mh, wedge_endpoints, wedge_midpoints
    )
    
    # JAX
    wdt_jax = cutils.get_wrapped_filtered_data_left_jax(
      quadrant, nbangles_perquad, jnp.array(x_hi), mv, mh, 
      tuple(wedge_endpoints), tuple(wedge_midpoints)
    )
    
    np.testing.assert_allclose(wdt_np, wdt_jax, atol=1e-7)

  def test_get_wedge_ticks_jax(self):
    """Verify get_wedge_ticks_jax matches NumPy version."""
    nbangles_perquad = 8
    m_horiz = 10.0
    
    # NumPy
    ticks_np = cutils.get_wedge_ticks(nbangles_perquad, m_horiz)
    
    # JAX
    ticks_jax = cutils.get_wedge_ticks_jax(nbangles_perquad, m_horiz)
    
    np.testing.assert_array_equal(ticks_np, ticks_jax)

  def test_get_wedge_end_mid_points_jax(self):
    """Verify get_wedge_end_mid_points_jax matches NumPy version."""
    nbangles_perquad = 8
    m_horiz = 10.0
    
    # NumPy
    ends_np, mids_np = cutils.get_wedge_end_mid_points(nbangles_perquad, m_horiz)
    
    # JAX
    ends_jax, mids_jax = cutils.get_wedge_end_mid_points_jax(nbangles_perquad, m_horiz)
    
    np.testing.assert_array_equal(ends_np, ends_jax)
    np.testing.assert_allclose(mids_np, mids_jax, atol=1e-12)

if __name__ == "__main__":
  absltest.main()
