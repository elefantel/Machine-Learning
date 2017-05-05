import numpy as np

#sigmoid derivative = sigmoid(1 - sigmoid)
def sigmoid_derivative(x):
    input = np.array(x)
    return input * (1 - input)