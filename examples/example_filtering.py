import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util
from fast_curvelet_transform.curvelet import fdct, ifdct, CurveletOptions

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
  """Apply soft thresholding to an array."""
  return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def run_filtering_example():
  """Demonstrate scale reconstruction and denoising."""
  print("--- Curvelet Filtering & Denoising Example ---")
  
  # 1. Load Image.
  image = data.camera().astype(np.float64) / 255.0
  n1, n2 = image.shape
  
  options = CurveletOptions(
    is_real=True,
    m=n1,
    n=n2,
    nbscales=5,
    nbangles_coarse=16,
    finest="wavelets",
    dtype=np.float64
  )
  
  # Forward Transform.
  coeffs = fdct(image, options)
  
  # 2. Scale-by-Scale Reconstruction.
  print("Visualizing reconstructions at each scale...")
  fig_scales, axes_scales = plt.subplots(1, 5, figsize=(20, 4))
  for j in range(5):
    # Create a copy with only one scale active.
    coeffs_scale = []
    for scale_idx in range(5):
      if scale_idx == j:
        coeffs_scale.append(coeffs[scale_idx])
      else:
        # Zero out other scales.
        coeffs_scale.append([np.zeros_like(a) for a in coeffs[scale_idx]])
    
    rec_scale = ifdct(coeffs_scale, options)
    axes_scales[j].imshow(rec_scale, cmap='gray')
    axes_scales[j].set_title(f"Scale {j}")
    axes_scales[j].axis('off')
  
  plt.tight_layout()
  plt.savefig('curvelet_scales.png')
  print("Saved scale visualization: curvelet_scales.png")
  
  # 3. Denoising via Soft Thresholding.
  print("Performing denoising demonstration...")
  sigma = 0.05
  noisy_image = image + sigma * np.random.randn(n1, n2)
  
  # Forward transform of noisy image.
  noisy_coeffs = fdct(noisy_image, options)
  
  # Apply thresholding.
  # Heuristic: threshold related to noise level.
  threshold = 0.5 * sigma 
  denoised_coeffs = []
  for scale in noisy_coeffs:
    denoised_scale = [soft_threshold(a, threshold) for a in scale]
    denoised_coeffs.append(denoised_scale)
  
  # Inverse transform.
  denoised_image = ifdct(denoised_coeffs, options)
  
  # Calculate PSNR improvement.
  psnr_noisy = 10 * np.log10(1 / np.mean((image - noisy_image)**2))
  psnr_denoised = 10 * np.log10(1 / np.mean((image - denoised_image)**2))
  print(f"Noisy PSNR: {psnr_noisy:.2f} dB")
  print(f"Denoised PSNR: {psnr_denoised:.2f} dB")
  
  # Visualization.
  fig_denoise, axes_denoise = plt.subplots(1, 3, figsize=(15, 5))
  axes_denoise[0].imshow(image, cmap='gray')
  axes_denoise[0].set_title("Original")
  axes_denoise[1].imshow(noisy_image, cmap='gray')
  axes_denoise[1].set_title(f"Noisy (PSNR={psnr_noisy:.1f}dB)")
  axes_denoise[2].imshow(denoised_image, cmap='gray')
  axes_denoise[2].set_title(f"Denoised (PSNR={psnr_denoised:.1f}dB)")
  
  for ax in axes_denoise:
    ax.axis('off')
    
  plt.tight_layout()
  plt.savefig('curvelet_denoising.png')
  print("Saved denoising visualization: curvelet_denoising.png")
  
  plt.show()

if __name__ == "__main__":
  run_filtering_example()
