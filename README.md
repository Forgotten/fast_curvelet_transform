# Fast Discrete Curvelet Transform (FDCT) in Python

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/fast_curvelet_transform/blob/main/notebooks/curvelet_demo.ipynb)

An efficient, double-precision implementation of the Fast Discrete Curvelet Transform (FDCT) developed by Candès, Demanet, Donoho, and Ying, using the wrapping approach. The implementation is ported from [Gabriel Peyré's MATLAB Toolbox](https://github.com/gpeyre/matlab-toolboxes), where the curvelet transform code was originally [written](https://github.com/gpeyre/matlab-toolboxes/blob/master/toolbox_wavelets/perform_curvelet_transform.m) by Laurent Demanet.

## Features

- **Machine Precision**: Implemented using `fp64` (double precision), achieving reconstruction errors $\approx 10^{-16}$.
- **Dataclass-based API**: Modern, type-safe configuration using `CurveletOptions`.
- **Comprehensive Support**: Supports both complex-valued and real-valued curvelets.
- **Pythonic Interface**: Clean API with support for standard NumPy arrays.
- **Verified**: Extensively tested for identity, isometry, adjoint property, and scale orthogonality.

## Installation

```bash
git clone https://github.com/Forgotten/fast_curvelet_transform.git
cd fast_curvelet_transform
pip install .
```

## Structure

- `fast_curvelet_transform/`: Core package containing the implementation.
  - `curvelet.py`: Forward and inverse transform logic.
- `examples/`: Demonstration scripts.
  - `example_cameraman.py`: Basic usage example.
  - `example_filtering.py`: Advanced scale reconstruction and denoising.
- `notebooks/`: Jupyter Notebook demos.
  - `curvelet_demo.ipynb`: Combined interactive demo (Colab friendly).
- `tests/`: Unit tests for verification.

## Usage

```python
import numpy as np
from fast_curvelet_transform.curvelet import fdct, ifdct, CurveletOptions

# Load or create an image
x = np.random.randn(512, 512).astype(np.float64)

# Setup options
options = CurveletOptions(is_real=True, m=512, n=512, nbscales=5)

# Forward Transform
c_coeffs = fdct(x, options)

# Inverse Transform
x_rec = ifdct(c_coeffs, options)

# Verify Reconstruction
print(f"Error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x)}")
```

## Running Tests

```bash
python3 -m unittest tests/test_curvelet.py
```

## References

- Candès, E., Demanet, L., Donoho, D., & Ying, L. (2006). Fast discrete curvelet transforms. *Multiscale Modeling & Simulation*, 5(3), 861-899.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
