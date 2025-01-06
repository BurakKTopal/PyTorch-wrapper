from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple_pytorch_wrapper", 
    version="0.0.6",
    author="Burakktopal",
    description="A lightweight PyTorch wrapper for fast and easy neural network training and evaluation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/burakktopal/pytorch-wrapper-package",
    packages=find_packages(include=["simple_pytorch_wrapper", "simple_pytorch_wrapper.*"]),
    include_package_data=True,  
    package_data={
        "simple_pytorch_wrapper.examples": ["data/*.npy"], 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "setuptools>=42.0.0"
    ],
)
