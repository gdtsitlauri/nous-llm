from setuptools import setup, find_packages

setup(
    name="nous-llm",
    version="0.1.0",
    description="NOUS: Neural Omnidirectional Understanding System — autonomous self-improving LLM framework",
    author="NOUS Contributors",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "llama-cpp-python>=0.2.56",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "networkx>=3.1",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
        "paper": ["matplotlib>=3.8.0", "pandas>=2.2.0"],
        "full": ["bitsandbytes>=0.43.0", "datasets>=2.18.0", "scipy>=1.11.0"],
    },
    entry_points={
        "console_scripts": [
            "nous=nous.cli:main",
        ],
    },
)
