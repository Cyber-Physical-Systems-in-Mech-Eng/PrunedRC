import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from benchmark_systems import load_data
from helpers_models import build_RC_model


"""
Lorenz system: 
input: [x(t), y[t), z(t)]], 
output: [x(t + 1), y(t + 1), z(t + 1)]
"""

# load data
x_train, y_train, x_test, y_test = load_data(name="lorenz", n_samples=10)
