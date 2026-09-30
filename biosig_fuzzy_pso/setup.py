from setuptools import setup, find_packages

setup(
    name="biosig_fuzzy_pso",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.0",
        "matplotlib>=3.7",
        "wfdb>=4.1",
    ],
    extras_require={"dev": ["pytest>=7.4"]},
)
