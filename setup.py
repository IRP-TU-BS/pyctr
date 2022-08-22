#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(name='continuum_robot_models',
      version='1.0',
      description='A Python package containing implementations of cosserat rod models for concentric tube continuum robots',
      author='Heiko Donat',
      author_email='h.donat@tu-bs.de',
      url='https://hdonat.net/pyctcr',
      packages=find_packages(exclude=('test*', 'docs')),
      zip_sage=False,
      install_requires=[
            'numpy>=1.14.5',
            'matplotlib>=2.2.0',
            'scipy',
            'matplotlib'
      ]
     )
