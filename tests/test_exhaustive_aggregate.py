import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform import curvelet_utils as cutils

jax.config.update("jax_enable_x64", True)

class TestExhaustiveAggregate(parameterized.TestCase):
  """Exhaustive unit tests for aggregate_from_wrapped_data_jax."""

  @parameterized.named_parameters(
    # Small sizes
    ("SmallRegular", 2, 2, 1, 1, "regular", None),
    ("SmallLeft", 2, 2, 1, 1, "left", None),
    ("SmallRight", 2, 2, 1, 1, "right", 5.0),
    
    # Medium sizes
    ("MediumRegular", 10, 20, 2, 5, "regular", None),
    ("MediumLeft", 10, 20, 2, 5, "left", None),
    ("MediumRight", 10, 20, 2, 5, "right", 15.0),
    
    # Large sizes
    ("LargeRegular", 50, 100, 10, 20, "regular", None),
    ("LargeLeft", 50, 100, 10, 20, "left", None),
    ("LargeRight", 50, 100, 10, 20, "right", 40.0),
    
    # Different shifts
    ("ShiftZeroRegular", 10, 10, 0, 0, "regular", None),
    ("ShiftLargeRegular", 10, 10, 50, 50, "regular", None),
  )
  def test_aggregate_exhaustive(self, lcw, ww, f_c, f_r, type_wedge, mh):
    """Exhaustively verify aggregate_from_wrapped_data_jax vs NumPy."""
    # Seed for reproducibility
    np.random.seed(42)
    
    # Create random input data
    wdata = np.random.randn(lcw, ww) + 1j * np.random.randn(lcw, ww)
    
    # Random l_l within a reasonable range to avoid massive xj
    # l_l determines column base. Let's say xj is large enough.
    xj_rows = lcw + f_r + 10
    xj_cols = ww + f_c + 20 + (int(2 * np.floor(4 * mh)) + 1 if mh else 0)
    
    # Ensure l_l starts at index >= 1 (since the code uses l_l[r]-1)
    l_l = np.random.randint(1, xj_cols - ww + 1, size=lcw)
    
    xj_np = np.zeros((xj_rows, xj_cols), dtype=complex)
    xj_jax = jnp.zeros((xj_rows, xj_cols), dtype=complex)
    
    # NumPy Reference
    out_np = cutils.aggregate_from_wrapped_data(
      wdata, xj_np, l_l, f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    # JAX Non-jitted
    out_jax = cutils.aggregate_from_wrapped_data_jax(
      jnp.array(wdata), xj_jax, jnp.array(l_l), f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    np.testing.assert_allclose(out_np, out_jax, atol=1e-12, rtol=1e-12)
    
    # JAX Jitted
    jitted_agg = jax.jit(
      cutils.aggregate_from_wrapped_data_jax,
      static_argnames=("f_c", "f_r", "lcw", "ww", "type_wedge", "mh")
    )
    out_jit = jitted_agg(
      jnp.array(wdata), xj_jax, jnp.array(l_l), f_c, f_r, lcw, ww, type_wedge=type_wedge, mh=mh
    )
    
    np.testing.assert_allclose(out_np, out_jit, atol=1e-12, rtol=1e-12)

if __name__ == "__main__":
  absltest.main()
