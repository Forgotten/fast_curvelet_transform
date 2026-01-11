import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform.curvelet import fdct_wrapping
from fast_curvelet_transform.curvelet_jax import fdct_wrapping_jax

jax.config.update("jax_enable_x64", True)

class TestCurveletJax(parameterized.TestCase):
  """Exhaustive tests for JAX Forward Curvelet Transform."""

  @parameterized.named_parameters(
    ("64_curvelets_complex", 64, "curvelets", False),
    ("64_wavelets_complex", 64, "wavelets", False),
    ("64_curvelets_real", 64, "curvelets", True),
    ("128_curvelets_complex", 128, "curvelets", False),
  )
  def test_fdct_wrapping_jax_parity(self, size, finest, is_real):
    """Verify fdct_wrapping_jax matches NumPy reference."""
    np.random.seed(size + (1 if is_real else 0))
    if is_real:
      x = np.random.randn(size, size)
    else:
      x = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    
    # NumPy Reference
    coeffs_np = fdct_wrapping(x, is_real=is_real, finest=finest)
    
    # JAX Version
    coeffs_jax = fdct_wrapping_jax(jnp.array(x), is_real=is_real, finest=finest)
    
    # Compare
    self.assertEqual(len(coeffs_np), len(coeffs_jax))
    for j in range(len(coeffs_np)):
      self.assertEqual(len(coeffs_np[j]), len(coeffs_jax[j]))
      for l in range(len(coeffs_np[j])):
        np.testing.assert_allclose(
          coeffs_np[j][l], 
          np.array(coeffs_jax[j][l]), 
          atol=1e-12, 
          rtol=1e-12,
          err_msg=f"Mismatch at scale {j}, angle {l}"
        )

  def test_fdct_wrapping_jax_jit(self):
    """Verify fdct_wrapping_jax can be JIT compiled."""
    size = 64
    x = jnp.array(np.random.randn(size, size) + 1j * np.random.randn(size, size))
    
    # JIT compile the transform
    # Note: finest and is_real are usually static or handled inside.
    # Here we just wrap it in a simple jit-able function if needed, 
    # but the internal utility calls are already jitted.
    # To jit the whole thing, we need to handle list of lists (pytree).
    
    @jax.jit
    def jitted_fdct(inp):
      return fdct_wrapping_jax(inp, is_real=False, finest="curvelets")
    
    coeffs = jitted_fdct(x)
    self.assertIsInstance(coeffs, list)
    self.assertIsInstance(coeffs[0], list)

if __name__ == "__main__":
  absltest.main()
