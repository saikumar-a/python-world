import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('ex1data1.txt',delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size


def plotData(x,y):
    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.show()
    
# plotData(X, y)

X = np.stack([np.ones(m), X], axis=1)

# print(X)