"""
Setup script

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from setuptools import setup

setup(
    name="pocket",
    version="0.5",
    description="A deep learning library to enable rapid prototyping.",
    author="Fred Zhang",
    author_email="frederic.zhang@anu.edu.au",
    packages=["pocket"],
    install_requires=[
        "torch>=1.5.1", "torchvision>=0.6.1",
        "matplotlib", "numpy", "scipy"
    ]
)
