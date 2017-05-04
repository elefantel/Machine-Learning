
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from sigmoid import sigmoid
from sigmoid import sigmoid_derivative

xdata = np.arange(-10.0, 10.0, 0.01)
sig = sigmoid(xdata)
sig_derivative = sigmoid_derivative(sig)

sigplot, = plt.plot(xdata, sig, 'r', label="Sigmoid")
sig_derivateive_plot, = plt.plot(xdata, sig_derivative, 'g', label="Sigmoid Derivative")
plt.legend(handler_map={sigplot: HandlerLine2D(numpoints=4)})
plt.grid()
plt.show()