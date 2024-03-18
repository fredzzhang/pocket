"""
The advanced visual package

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
"""

try:
    import seaborn
except ImportError:
    raise ImportError(
        "pocket.advis requires the package seaborn. "
        "Please run pip install seaborn."
    )

from .colours import *
from .heatmap import *
from .text import *
from .ellipse import *
