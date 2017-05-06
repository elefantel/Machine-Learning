import numpy as np
import math

#sigmoid = 1 / (1 + e^-x)
#Used to normalize values
def sigmoid(x, derivative=False):

    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))