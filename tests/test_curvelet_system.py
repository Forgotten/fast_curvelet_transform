import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from fast_curvelet_transform import (
    get_curvelet_system,
    CurveletOptions,
    CurveletSystem
)

class TestCurveletSystem(parameterized.TestCase):
    """Unit tests for the CurveletSystem Pytree."""

    @parameterized.product(
        shape=[(32, 32), (64, 32)],
        is_real=[True, False],
        finest=["wavelets", "curvelets"]
    )
    def test_get_curvelet_system_jit(self, shape, is_real, finest):
        """Verify that get_curvelet_system works and returns a JIT-compatible Pytree."""
        m, n = shape
        options = CurveletOptions(
            is_real=is_real,
            m=m,
            n=n,
            nbscales=3,
            finest=finest,
            dtype=np.complex128 if not is_real else np.float64
        )

        # Generate system.
        system = get_curvelet_system(m, n, options)

        self.assertIsInstance(system, CurveletSystem)
        self.assertIsInstance(system.waveforms, jnp.ndarray)
        self.assertIsInstance(system.dimensions, jnp.ndarray)

        # Verify JIT compatibility.
        @jax.jit
        def get_waveforms(s):
            return s.waveforms

        waveforms_jit = get_waveforms(system)
        np.testing.assert_allclose(waveforms_jit, system.waveforms)

    def test_dimensions_consistency(self):
        """Verify that the stored dimensions match the waveforms count."""
        m, n = 32, 32
        options = CurveletOptions(m=m, n=n, nbscales=3)
        system = get_curvelet_system(m, n, options)

        # Total number of curvelets should match the number of waveforms.
        self.assertEqual(len(system.waveforms), len(system.dimensions))
        
        # Verify that each waveform has the correct image size.
        for waveform in system.waveforms:
            self.assertEqual(waveform.shape, (32, 32))

    def test_jax_transforms_reconstruction(self):
        """Verify that CurveletSystem.jax_fdct_2d and inverse are consistent."""
        m, n = 32, 32
        options = CurveletOptions(m=m, n=n, nbscales=3)
        system = get_curvelet_system(m, n, options)

        # Create a random image.
        key = jax.random.PRNGKey(0)
        img = jax.random.normal(key, (m, n))

        # Forward transform.
        coeffs = system.jax_fdct_2d(img)
        
        # Inverse transform and sum.
        img_reconstructed = system.reconstruct(coeffs)

        # Check reconstruction error.
        error = jnp.linalg.norm(img - img_reconstructed) / jnp.linalg.norm(img)
        self.assertLess(error, 1e-5)

    def test_differentiability(self):
        """Verify that the transform is differentiable with respect to the input image."""
        m, n = 32, 32
        options = CurveletOptions(m=m, n=n, nbscales=3)
        system = get_curvelet_system(m, n, options)

        # Create a random image.
        key = jax.random.PRNGKey(1)
        img = jax.random.normal(key, (m, n))

        def loss_fn(x):
            coeffs = system.jax_fdct_2d(x)
            return jnp.sum(jnp.abs(coeffs)**2)

        # Compute gradient.
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(img)

        self.assertEqual(grad.shape, img.shape)
        self.assertFalse(jnp.all(grad == 0))

if __name__ == '__main__':
    absltest.main()
