"""
Add ${POCKET} to Python system path

Written by Frederic Zhang
Australian National University

Lasted updated in Mar. 2019
"""

import os
import sys

sys.path.insert(0, os.environ['POCKET'])
