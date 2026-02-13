"""
Setup script for Bank Marketing EDA Project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bank-marketing-eda",
    version="0.1.0",
    description="Comprehensive EDA for UCI Bank Marketing Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Final Project Team",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-eda=src.eda.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
