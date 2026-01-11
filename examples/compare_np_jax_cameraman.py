import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skimage import data
import time
from fast_curvelet_transform.curvelet import fdct_wrapping, ifdct_wrapping
from fast_curvelet_transform.curvelet_jax import fdct_wrapping_jax

jax.config.update("jax_enable_x64", True)

def run_comparison():
  """Compare NumPy and JAX versions of the Forward Curvelet Transform."""
  print("--- Fast Curvelet Transform: NumPy vs JAX Comparison ---")
  
  # 1. Load data
  image = data.camera().astype(np.float64)
  n1, n2 = image.shape
  print(f"Loaded image of size {n1}x{n2}")
  
  # Parameters
  is_real = True
  finest = "curvelets"
  nbscales = 5
  nbangles_coarse = 16
  
  # 2. Forward Transform: NumPy
  print("\n[NumPy] Benchmarking forward transform (5 trials)...")
  np_times = []
  for i in range(5):
    start_np = time.time()
    coeffs_np = fdct_wrapping(
      image, is_real=is_real, finest=finest, nbscales=nbscales, nbangles_coarse=nbangles_coarse
    )
    np_times.append(time.time() - start_np)
  
  print(f"NumPy Mean: {np.mean(np_times):.4f}s, Variance: {np.var(np_times):.2e}s")
  
  # 3. Forward Transform: JAX
  print("\n[JAX] Computing first call (including JIT)...")
  image_jax = jnp.array(image)
  start_jax = time.time()
  coeffs_jax = fdct_wrapping_jax(
    image_jax, is_real=is_real, finest=finest, nbscales=nbscales, nbangles_coarse=nbangles_coarse
  )
  jax.block_until_ready(coeffs_jax)
  print(f"JAX First Call took: {time.time() - start_jax:.4f}s")
  
  print("\n[JAX] Benchmarking warm forward transform (5 trials)...")
  jax_times = []
  for i in range(5):
    start_jax = time.time()
    coeffs_jax = fdct_wrapping_jax(
      image_jax, is_real=is_real, finest=finest, nbscales=nbscales, nbangles_coarse=nbangles_coarse
    )
    jax.block_until_ready(coeffs_jax)
    jax_times.append(time.time() - start_jax)
  
  print(f"JAX Warm Mean: {np.mean(jax_times):.4f}s, Variance: {np.var(jax_times):.2e}s")
  
  # 4. Compare Coefficients
  print("\nComparing coefficients...")
  max_diff_coeffs = 0
  for j in range(len(coeffs_np)):
    for l in range(len(coeffs_np[j])):
      diff = np.abs(coeffs_np[j][l] - np.array(coeffs_jax[j][l]))
      max_diff_coeffs = max(max_diff_coeffs, np.max(diff))
  print(f"Maximum difference in coefficients: {max_diff_coeffs:.2e}")
  
  # 5. Inverse Transform: NumPy (on both coefficients)
  print("\n[NumPy] Computing inverse transform from NumPy coefficients...")
  rec_np = ifdct_wrapping(
    coeffs_np, is_real=is_real, m_img=n1, n_img=n2
  )
  
  print("[NumPy] Computing inverse transform from JAX coefficients...")
  # Convert JAX coeffs to NumPy for the inverse function
  coeffs_jax_np = [[np.array(c) for c in scale] for scale in coeffs_jax]
  rec_jax = ifdct_wrapping(
    coeffs_jax_np, is_real=is_real, m_img=n1, n_img=n2
  )
  
  # 6. Compare Reconstructions
  error_np = np.linalg.norm(image - rec_np) / np.linalg.norm(image)
  error_jax = np.linalg.norm(image - rec_jax) / np.linalg.norm(image)
  rec_diff = np.linalg.norm(rec_np - rec_jax) / np.linalg.norm(rec_np)
  
  print(f"\nReconstruction error (NP coeffs): {error_np:.2e}")
  print(f"Reconstruction error (JAX coeffs): {error_jax:.2e}")
  print(f"Relative difference between reconstructions: {rec_diff:.2e}")
  
  # 7. Visualization
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  axes[0].imshow(np.real(image), cmap='gray')
  axes[0].set_title("Original Image")
  axes[0].axis('off')
  
  axes[1].imshow(np.real(rec_jax), cmap='gray')
  axes[1].set_title("Reconstructed (from JAX coeffs)")
  axes[1].axis('off')
  
  diff_map = np.abs(np.real(rec_np) - np.real(rec_jax))
  im = axes[2].imshow(diff_map, cmap='hot')
  axes[2].set_title("Abs Difference (NP vs JAX Rec)")
  axes[2].axis('off')
  plt.colorbar(im, ax=axes[2])
  
  plt.tight_layout()
  plt.savefig('np_jax_comparison.png')
  print("\nArtifact saved: np_jax_comparison.png")
  plt.show()

if __name__ == "__main__":
  run_comparison()
