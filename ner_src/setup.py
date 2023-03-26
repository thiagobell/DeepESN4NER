#!/usr/bin/env python
from setuptools import find_packages, setup


setup(name="deepesn_ner",
      version='0.1',
      packages=find_packages(),
      install_requires=["torch", "flair", "psutil", "sklearn", "pathos"])
