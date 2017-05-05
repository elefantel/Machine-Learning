import math

#sigmoid = 1 / (1 + e^-x)
def sigmoid(x):
    data = []

    for item in x:
        data.append(1 / (1 + math.exp(-item)))
    return data