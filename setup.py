from setuptools import setup, find_packages

setup(
    name="kerneltracer",
    version="0.1.0",
    description="Unified kernel tracing framework",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "tqdm",
        "rich",
        "psutil",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
            "mypy",
        ],
        "vllm": [
            "vllm>=0.11.0",
        ],
    },
)
