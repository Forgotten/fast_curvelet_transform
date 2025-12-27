from setuptools import setup, find_packages

setup(
    name="fast_curvelet_transform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
    ],
    author="Leonardo Zepeda-Núñez",
    author_email="lzepeda@melix.org",
    description="A fast discrete curvelet transform implementation in Python.",
    keywords="curvelets, wavelet, signal processing, image processing",
    python_requires=">=3.10",
    url="https://github.com/Forgotten/fast_curvelet_transform",
    license="MIT",
)
