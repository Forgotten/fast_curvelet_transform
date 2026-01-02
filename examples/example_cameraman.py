import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import time
from fast_curvelet_transform.curvelet import fdct, ifdct, CurveletOptions

def run_example():
  """Run a demonstration of the Curvelet Transform on the cameraman image."""
  print("--- Fast Curvelet Transform Example ---")
  
  # Load cameraman image.
  image = data.camera().astype(np.float64)
  n1, n2 = image.shape
  print(f"Loaded image of size {n1}x{n2}")
  
  # Setup options.
  options = CurveletOptions(
    is_real=True,
    m=n1,
    n=n2,
    nbscales=6,
    nbangles_coarse=16,
    finest="wavelets",
    dtype=np.float64
  )
  
  # 1. Forward Transform.
  print("Computing forward transform...")
  start_time = time.time()
  c_coeffs = fdct(image, options)
  print(f"Forward transform took: {time.time() - start_time:.2f} seconds")
  # 2. Analyze coefficients.
  total_elements = sum(a.size for scale in c_coeffs for a in scale)
  print(f"Redundancy factor: {total_elements / image.size:.2f}")
  
  # 3. Inverse Transform.
  print("Computing inverse transform...")
  start_time = time.time()
  recovered = ifdct(c_coeffs, options)
  print(f"Inverse transform took: {time.time() - start_time:.2f} seconds")
  
  # 4. Verification.
  error = np.linalg.norm(image - recovered) / np.linalg.norm(image)
  print(f"Reconstruction Error (Relative L2): {error:.2e}")
  
  # 5. Visualization.
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  axes[0].imshow(image, cmap='gray')
  axes[0].set_title("Original Image")
  axes[0].axis('off')
  
  axes[1].imshow(recovered, cmap='gray')
  axes[1].set_title("Reconstructed Image")
  axes[1].axis('off')
  
  error_map = np.abs(image - recovered)
  im_err = axes[2].imshow(error_map, cmap='hot')
  axes[2].set_title("Absolute Error Map")
  axes[2].axis('off')
  plt.colorbar(im_err, ax=axes[2])
  
  plt.tight_layout()
  plt.savefig('curvelet_reconstruction.png')
  print("Artifact saved: curvelet_reconstruction.png")
  plt.show()

if __name__ == "__main__":
  run_example()
