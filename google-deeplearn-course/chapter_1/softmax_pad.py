import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#plot scores
import  matplotlib.pyplot as plt

x = np.arange(-2, 6, 0.1) #scores
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
#multiplying the scores by a large number makes the softmax output close to either 0 or 1
#i.e increasing the size of the outputs, the classifier becomes very confident of its predictions
#dividing the scores by a large number makes the softmax output close to a uniform distribution

plt.plot(x, softmax(scores).T, linewidth=2)
plt.xlabel("x")
plt.ylabel("softmax")
plt.grid()
plt.show()