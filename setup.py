from setuptools import find_packages, setup

setup(
    name="neural-fx",
    version="0.1.0",
    python_requires=">=3.10,<3.14",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "torch>=2.11,<2.12",
        "torchaudio>=2.11,<2.12",
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "lightning",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest>=8,<10",
            "pytest-cov>=6,<8",
            "pytest-xdist>=3,<4",
            "ruff>=0.11,<0.17",
        ],
    },
)
