import math
import numpy as np

#sigmoid = 1 / (1 + e^-x)
def sigmoid(x):
    data = []

    for item in x:
        data.append(1 / (1 + math.exp(-item)))
    return data

#sigmoid derivative = sigmoid(1 - sigmoid)
def sigmoid_derivative(x):
    xarray = np.array(x)
    return xarray * (1 - xarray)
