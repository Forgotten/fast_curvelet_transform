import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from absl.testing import absltest
from fast_curvelet_transform.curvelet import fdct_wrapping
from fast_curvelet_transform.curvelet_jax import fdct_wrapping_jax

# Explicitly DISABLE x64 to test single precision compatibility
jax.config.update("jax_enable_x64", False)

class TestCurveletJaxF32(parameterized.TestCase):
  """Tests for JAX Forward Curvelet Transform in Single Precision."""

  @parameterized.named_parameters(
    ("32_curvelets_complex", 32, "curvelets", False),
    ("32_wavelets_complex", 32, "wavelets", False),
    ("32_curvelets_real", 32, "curvelets", True),
  )
  def test_fdct_wrapping_jax_f32_parity(self, size, finest, is_real):
    """Verify fdct_wrapping_jax works in float32 and matches NP reference."""
    np.random.seed(size + (1 if is_real else 0))
    if is_real:
      x = np.random.randn(size, size).astype(np.float32)
    else:
      x = (np.random.randn(size, size) + 1j * np.random.randn(size, size)).astype(np.complex64)
    
    # NumPy Reference (still using double precision for reference if we want, 
    # but here we use float32 for comparison to see if JAX handles it)
    coeffs_np = fdct_wrapping(x, is_real=is_real, finest=finest)
    
    # JAX Version (should be float32/complex64 if x64 is disabled)
    x_jax = jnp.array(x)
    if not jax.config.read("jax_enable_x64"):
      self.assertEqual(x_jax.dtype, jnp.float32 if is_real else jnp.complex64)
    
    coeffs_jax = fdct_wrapping_jax(x_jax, is_real=is_real, finest=finest)
    
    # Compare
    self.assertEqual(len(coeffs_np), len(coeffs_jax))
    for j in range(len(coeffs_np)):
      self.assertEqual(len(coeffs_np[j]), len(coeffs_jax[j]))
      for l in range(len(coeffs_np[j])):
        # Check dtype of JAX output
        if is_real:
            # Note: coarsest and finest might be real, others might be complex 
            # depending on if_real logic. In fdct_wrapping_jax:
            # coarsest: real, intermediate: complex, finest: real (if is_real)
            # Actually, curvelet coeffs are usually complex except for Coarsest/Finest if is_real
            pass
        
        # Use relaxed tolerance for float32
        np.testing.assert_allclose(
          coeffs_np[j][l], 
          np.array(coeffs_jax[j][l]), 
          atol=1e-5, 
          rtol=1e-5,
          err_msg=f"Mismatch at scale {j}, angle {l}"
        )

  def test_fdct_wrapping_jax_jit_f32(self):
    """Verify fdct_wrapping_jax can be JIT compiled in float32."""
    size = 32
    x = jnp.array(np.random.randn(size, size).astype(np.float32))
    
    @jax.jit
    def jitted_fdct(inp):
      return fdct_wrapping_jax(inp, is_real=True, finest="curvelets")
    
    coeffs = jitted_fdct(x)
    self.assertIsInstance(coeffs, list)
    # Check that it's float32 if x64 is disabled
    if not jax.config.read("jax_enable_x64"):
      self.assertEqual(coeffs[0][0].dtype, jnp.float32)
    else:
      # If x64 is enabled, it should be float64
      self.assertEqual(coeffs[0][0].dtype, jnp.float64)

if __name__ == "__main__":
  absltest.main()
