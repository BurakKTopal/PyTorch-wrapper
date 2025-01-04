from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="burakktopal-test-pytorch-wrapper-package",
    version="0.0.1",
    author="Burakktopal",
    # author_email="burak.kucuktopal@gmail.com",
    description="A lightweight PyTorch wrapper for fast and easy neural network training and evaluation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/burakktopal/pytorch-wrapper-package",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "examples": ["data/*.npy"],  # Include .npy files from the 'data' subfolder of 'examples'
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
        "ipython>=7.0.0",
    ],
)
