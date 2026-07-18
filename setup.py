from setuptools import setup, find_packages

setup(
    name="neural-fx",
    version="0.1",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"neuralfx": "neural_fx"},
    include_package_data=True,
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "lightning",
        "tensorboard",
        "pyyaml",
    ],
    extras_require={"dev": ["pytest", "pytest-cov", "pytest-xdist", "ruff"]},
)
