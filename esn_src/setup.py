#!/usr/bin/env python
from setuptools import find_packages, setup


setup(name="esn_toolkit",
      version='0.1',
      packages=find_packages(),
      install_requires=["torch", "numpy", "pytest", "matplotlib", "p_tqdm"])
