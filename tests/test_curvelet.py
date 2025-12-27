import numpy as np
import pytest
from fast_curvelet_transform.curvelet import (
  fdct,
  ifdct,
  CurveletOptions,
  fdct_wrapping_window
)

def test_window_properties():
  """Verify partition of unity and symmetry of the curvelet window."""
  x = np.linspace(0, 1, 100)
  wl, wr = fdct_wrapping_window(x)
  
  # Check partition of unity: wl^2 + wr^2 = 1.
  np.testing.assert_allclose(wl**2 + wr**2, 1.0, atol=1e-12)
  
  # Check symmetry.
  np.testing.assert_allclose(wl, wr[::-1], atol=1e-12)
  
  # Check boundaries.
  assert np.isclose(wl[0], 0, atol=1e-12)
  assert np.isclose(wr[0], 1, atol=1e-12)
  assert np.isclose(wl[-1], 1, atol=1e-12)
  assert np.isclose(wr[-1], 0, atol=1e-11)

@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (64, 32)])
@pytest.mark.parametrize("is_real", [True, False])
@pytest.mark.parametrize("finest", ["wavelets", "curvelets"])
def test_identity(shape, is_real, finest):
  """Verify that ifdct(fdct(x)) == x for multiple configurations."""
  m, n = shape
  if is_real:
    x = np.random.randn(m, n).astype(np.float64)
  else:
    x = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(np.complex128)
  
  # Setup options.
  dtype = np.complex128 if not is_real else np.float64
  options = CurveletOptions(
    is_real=is_real,
    m=m,
    n=n,
    nbscales=3,
    finest=finest,
    dtype=dtype
  )
  
  # Forward Transform.
  c_coeffs = fdct(x, options)
  
  # Check dtype of coefficients.
  for scale in c_coeffs:
    for angle in scale:
      assert angle.dtype == dtype
  
  # Inverse Transform.
  x_rec = ifdct(c_coeffs, options)
  
  # Check dtype of reconstruction.
  assert x_rec.dtype == dtype
  
  # Verify reconstruction error.
  error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
  assert error < 1e-10

@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (64, 32)])
@pytest.mark.parametrize("is_real", [True, False])
@pytest.mark.parametrize("finest", ["wavelets", "curvelets"])
def test_energy_preservation(shape, is_real, finest):
  """Verify isometry (Parseval/Plancherel) of the curvelet transform."""
  m, n = shape
  if is_real:
    x = np.random.randn(m, n).astype(np.float64)
  else:
    x = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(np.complex128)
  
  # Setup options.
  options = CurveletOptions(
    is_real=is_real,
    m=m,
    n=n,
    nbscales=3,
    finest=finest
  )
  
  # Forward Transform.
  c_coeffs = fdct(x, options)
  
  # Compute energy in pixel domain.
  energy_x = np.sum(np.abs(x)**2)
  
  # Compute energy in curvelet domain.
  energy_c = sum(
    np.sum(np.abs(angle)**2) for scale in c_coeffs for angle in scale
  )
      
  # Check if ratio is 1.0.
  ratio = energy_c / energy_x
  np.testing.assert_allclose(ratio, 1.0, atol=1e-10)

@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("is_real", [True, False])
def test_scale_orthogonality(shape, is_real):
  """Verify that energy is preserved per scale and check cross-scale overlap."""
  m, n = shape
  if is_real:
    x = np.random.randn(m, n).astype(np.float64)
  else:
    x = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(np.complex128)
  
  options = CurveletOptions(is_real=is_real, m=m, n=n, nbscales=4)
  c_coeffs = fdct(x, options)
  
  # Reconstruct each scale separately.
  reconstructions = []
  for j in range(len(c_coeffs)):
    c_single = []
    for s_idx in range(len(c_coeffs)):
      if s_idx == j:
        c_single.append(c_coeffs[s_idx])
      else:
        c_single.append([np.zeros_like(a) for a in c_coeffs[s_idx]])
    reconstructions.append(ifdct(c_single, options))
  
  # Check energy consistency: Sum of |c_j|^2 matches |f|^2.
  energy_sum_c = sum(np.sum(np.abs(a)**2) for s in c_coeffs for a in s)
  energy_x = np.sum(np.abs(x)**2)
  np.testing.assert_allclose(energy_sum_c, energy_x, rtol=1e-10)

  # Check orthogonality: non-adjacent scales should be perfectly orthogonal.
  # Adjacent scales overlap by design in the FDCT (error ~0.05).
  for j in range(len(reconstructions)):
    for k in range(j + 2, len(reconstructions)):
        inner = np.abs(np.sum(reconstructions[j] * np.conj(reconstructions[k])))
        norm_j = np.linalg.norm(reconstructions[j])
        norm_k = np.linalg.norm(reconstructions[k])
        if norm_j > 1e-12 and norm_k > 1e-12:
            assert inner / (norm_j * norm_k) < 1e-10

@pytest.mark.parametrize("shape", [(32, 32), (64, 64)])
@pytest.mark.parametrize("is_real", [True, False])
@pytest.mark.parametrize("finest", ["wavelets", "curvelets"])
def test_adjoint(shape, is_real, finest):
  """Verify the adjoint property: <Tf, c> = <f, T*c>."""
  m, n = shape
  if is_real:
    f = np.random.randn(m, n).astype(np.float64)
  else:
    f = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(np.complex128)
  
  options = CurveletOptions(is_real=is_real, m=m, n=n, nbscales=3, finest=finest)
  c_base = fdct(f, options)
  
  # Create random coefficients c matching the FDCT structure.
  c = []
  dtype = np.complex128 if not is_real else np.float64
  for scale in c_base:
    c_scale = []
    for angle in scale:
      if is_real:
        c_scale.append(np.random.randn(*angle.shape).astype(dtype))
      else:
        c_scale.append(
          (np.random.randn(*angle.shape) + 1j * np.random.randn(*angle.shape)).astype(dtype)
        )
    c.append(c_scale)
  
  Tf = fdct(f, options)
  T_star_c = ifdct(c, options)
  
  # <Tf, c> in coefficient space.
  dot_Tf_c = sum(np.sum(Tf[j][l] * np.conj(c[j][l])) for j in range(len(c)) for l in range(len(c[j])))
  
  # <f, T*c> in image space.
  dot_f_Tstar_c = np.sum(f * np.conj(T_star_c))
  
  np.testing.assert_allclose(dot_Tf_c, dot_f_Tstar_c, rtol=1e-10, atol=1e-10)
