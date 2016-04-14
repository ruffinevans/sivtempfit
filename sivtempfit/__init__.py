# This file is automatically run when the package is imported
# So, a few key definitions can be placed here.
# It should import things from other .py files in this directory, e.g.
# from .text import test_asdf()
# See https://python-packaging.readthedocs.org/en/latest/minimal.html

import numpy as np
import seaborn as sns
import emcee
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats  # Use for KDE in the end
import scipy.optimize as opt  # Use for minimizing KDE
import dataprocessing as dp


def test_asdf():
    return 1
