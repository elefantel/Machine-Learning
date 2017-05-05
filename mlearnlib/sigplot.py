import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from mlearnlib import sigmoid as sg
from mlearnlib import sigmoid_derivative as sgd

xdata = np.arange(-10.0, 10.0, 0.01)
sigfunc = sg.sigmoid(xdata) #sigmoid(xdata)
sig_derivative = sgd.sigmoid_derivative(sigfunc)

sigplot, = plt.plot(xdata, sigfunc, 'r', label="Sigmoid")
sig_derivateive_plot, = plt.plot(xdata, sig_derivative, 'g', label="Sigmoid Derivative")
plt.legend(handler_map={sigplot: HandlerLine2D(numpoints=4)})
plt.grid()
plt.show()