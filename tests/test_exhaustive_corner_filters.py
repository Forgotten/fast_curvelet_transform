import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform import curvelet_utils as cutils

jax.config.update("jax_enable_x64", True)

class TestExhaustiveCornerFilters(parameterized.TestCase):
  """Exhaustive unit tests for corner filter JAX functions."""

  @parameterized.named_parameters(
    ("8_Q1_M10", 8, 1, 10.0, 10.0),
    ("8_Q2_M15", 8, 2, 15.0, 15.0),
    ("16_Q3_M20", 16, 3, 20.0, 20.0),
    ("4_Q4_M5", 4, 4, 5.0, 5.0),
    ("8_Q2_M40", 8, 2, 40.0, 40.0),
  )
  def test_get_wrapped_filtered_data_right_exhaustive(self, nbangles_perquad, quadrant, mv, mh):
    """Verify get_wrapped_filtered_data_right_jax vs NumPy reference."""
    size = 2 * int(np.floor(4 * max(mv, mh))) + 1
    np.random.seed(nbangles_perquad + quadrant + int(mv))
    x_hi = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    f_r = 5
    
    # NumPy
    out_np = cutils.get_wrapped_filtered_data_right(
      quadrant, nbangles_perquad, x_hi, f_r, mv, mh, wedge_endpoints, wedge_midpoints
    )
    
    # JAX
    out_jax = cutils.get_wrapped_filtered_data_right_jax(
      quadrant, nbangles_perquad, jnp.array(x_hi), f_r, mv, mh, 
      tuple(wedge_endpoints), tuple(wedge_midpoints)
    )
    
    np.testing.assert_allclose(out_np, out_jax, atol=1e-12, rtol=1e-12)

  @parameterized.named_parameters(
    ("8_Q1_M10", 8, 1, 10.0, 10.0),
    ("8_Q2_M15", 8, 2, 15.0, 15.0),
    ("16_Q3_M20", 16, 3, 20.0, 20.0),
    ("4_Q4_M5", 4, 4, 5.0, 5.0),
    ("8_Q2_M40", 8, 2, 40.0, 40.0),
  )
  def test_get_wrapped_filtered_data_left_exhaustive(self, nbangles_perquad, quadrant, mv, mh):
    """Verify get_wrapped_filtered_data_left_jax vs NumPy reference."""
    size = 2 * int(np.floor(4 * max(mv, mh))) + 1
    np.random.seed(nbangles_perquad + quadrant + int(mv) + 100)
    x_hi = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    wedge_endpoints, wedge_midpoints = cutils.get_wedge_end_mid_points(nbangles_perquad, mh)
    
    # NumPy
    out_np = cutils.get_wrapped_filtered_data_left(
      quadrant, nbangles_perquad, x_hi, mv, mh, wedge_endpoints, wedge_midpoints
    )
    
    # JAX
    out_jax = cutils.get_wrapped_filtered_data_left_jax(
      quadrant, nbangles_perquad, jnp.array(x_hi), mv, mh, 
      tuple(wedge_endpoints), tuple(wedge_midpoints)
    )
    
    np.testing.assert_allclose(out_np, out_jax, atol=1e-12, rtol=1e-12)

if __name__ == "__main__":
  absltest.main()
