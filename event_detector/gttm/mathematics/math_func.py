import numpy as np

"""
http://xaktly.com/LogisticFunctions.html
"""


def sigmoid(x, x0=0, a=1, b=1):
    return a / (1 + b * np.exp(x0 - x))


def linear(x, x_max):
    return x / x_max
