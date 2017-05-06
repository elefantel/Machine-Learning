#Testing Sigmoid function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from mlearnlib import sigmoid as sg

input = np.arange(-10.0, 10.0, 0.01)

sigfunc = sg.sigmoid(input)

print(sigfunc)
sig_derivative = sg.sigmoid(sigfunc, derivative=True) #Sigmoid derivative

sigplot, = plt.plot(input, sigfunc, 'r', label="Sigmoid")
print(sigfunc)
sig_derivative_plot, = plt.plot(input, sig_derivative, 'g', label="Sigmoid Derivative")
print(sig_derivative)
plt.legend(handler_map={sigplot: HandlerLine2D(numpoints=4)})
plt.grid()
plt.show()