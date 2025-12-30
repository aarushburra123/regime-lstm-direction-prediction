from setuptools import setup, find_packages

setup(
    name="regime-trader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",

    ],
    python_requires=">=3.10",
    author="Aarush Burra",
    description="Regime-aware quantitative trading toolkit",
)
