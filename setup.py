from setuptools import find_packages, setup

setup(
    name="fame",
    version="0.1.0",
    description="Feature Activation Map Explanation (CVPRW 2026)",
    packages=find_packages(include=["fame", "fame.*"]),
    python_requires=">=3.9",
)
