import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import time
from fast_curvelet_transform import (
    get_curvelet_system,
    CurveletOptions
)

jax.config.update('jax_enable_x64', True)

def run_jax_example():
    """Run a demonstration of the JAX Curvelet Transform implementation."""
    print("--- JAX Curvelet Transform Example (Cameraman) ---")
    
    # Load cameraman image and convert to JAX array.
    image_np = data.camera().astype(np.float32) / 255.0
    # Downsample for speed.
    image_np = image_np
    image = jnp.array(image_np)
    m, n = image.shape
    print(f"Loaded image of size {m}x{n}")
    
    # Setup Curvelet Options.
    options = CurveletOptions(
        is_real=True,
        m=m,
        n=n,
        nbscales=4,
        nbangles_coarse=16,
        finest="curvelets",
        dtype=np.complex64
    )
    
    # Generate the Curvelet System (Pre-computation).
    print("Generating Curvelet System (Pre-computation)...")
    start_time = time.time()
    system = get_curvelet_system(m, n, options)
    print(f"System generation took: {time.time() - start_time:.2f} seconds")
    
    # Prepare support sizes for inverse transform.
    support_sizes = jnp.prod(system.dimensions, axis=1)
    
    # Define JIT-compiled transform functions for speed.
    @jax.jit
    def forward_transform(img, system):
        return system.jax_fdct_2d(img)
    
    @jax.jit
    def inverse_transform(coeffs, system):
        return system.reconstruct(coeffs)

    # Warm up JIT.
    print("Warming up JIT...")
    _ = forward_transform(image, system)
    _ = inverse_transform(jnp.zeros((len(system.waveforms), m, n), dtype=image.dtype), 
                          system)
    
    # Perform Forward Transform.
    print("Computing Forward Transform (JAX JIT)...")
    start_time = time.time()
    coeffs = forward_transform(image, system)
    # Trigger execution for timing.
    coeffs.block_until_ready()
    print(f"Forward transform took: {time.time() - start_time:.4f} seconds")
    
    # Perform Inverse Transform (Reconstruction).
    print("Computing Inverse Transform (JAX JIT)...")
    start_time = time.time()
    reconstructed = inverse_transform(coeffs, system)
    reconstructed.block_until_ready()
    print(f"Inverse transform took: {time.time() - start_time:.4f} seconds")
    
    # Verification.
    # Note: For real signals, we take the real part.
    reconstructed_real = jnp.real(reconstructed)
    error = jnp.linalg.norm(image - reconstructed_real) / jnp.linalg.norm(image)
    print(f"Reconstruction Error (Relative L2): {error:.2e}")
    
    # Visualization.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Use real part for display.
    axes[1].imshow(np.array(reconstructed_real), cmap='gray')
    axes[1].set_title("Reconstructed (JAX)")
    axes[1].axis('off')
    
    error_map = np.abs(image_np - np.array(reconstructed_real))
    im_err = axes[2].imshow(error_map, cmap='hot')
    axes[2].set_title("Absolute Error Map")
    axes[2].axis('off')
    plt.colorbar(im_err, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('curvelet_jax_reconstruction.png')
    print("Artifact saved: curvelet_jax_reconstruction.png")
    # plt.show() # Disabled for headless execution

if __name__ == "__main__":
    run_jax_example()
