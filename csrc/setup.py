"""
Setup script for c++ extensions

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            name='cpp',
            sources=['masks.cpp']
        )],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
