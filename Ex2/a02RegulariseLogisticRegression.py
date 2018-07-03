import numpy as np
from matplotlib import pyplot
from scipy import optimize
from Ex2.utils import plotData, mapFeature, plotDecisionBoundary

data=np.loadtxt('ex2data2.txt',delimiter=',')
X, y = data[:, :2], data[:, 2]

plotData(X,y)
# Labels and Legend
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
# Specified in plot order
pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
# pyplot.show()

# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X[:, 0], X[:, 1])
# print(X)

def h(X, theta):
    """compute h(x) = sigmoid(X*theta)
       this is the matrix version, this simultaneously multiplies for all feature rows
    """
    theta_t_x = np.matmul(X, theta)
    return sigmoid(theta_t_x)

def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    return 1 / (1 +  np.e**(-z))

def predict(theta, X):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    h_vec = h(X, theta)
    #for iindex, i in enumerate(h_mat):
    
    for iindex, i in np.ndenumerate(h_vec):
        p[iindex[0]] = 0 if i < 0.5 else 1
    
    # ============================================================
    return p


def costFunctionReg(theta, X, y, lambda_):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================
    J_inner = np.matmul(np.log(h(X, theta)), -y) - np.matmul(np.log(1-h(X, theta)), (1-y))  # copied from above
    J_first_term = (1/m) * np.sum(J_inner)
    J_second_term = (lambda_/(2*m))* np.sum(theta[1:]**2)  # note the 1:, j goes from 1 not 0
    J = J_first_term + J_second_term

    # COMPUTE GRADIENT
    # FIRST PART OF THIS COPIED FROM ABOVE. Second term is new. 
    grad_second_term = np.copy(theta)
    grad_second_term[0] = 0
    grad = (1/m) * np.matmul((h(X, theta) - y), X) + (lambda_/m)*grad_second_term
    # =============================================================
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')


# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# set options for optimize.minimize
options= {'maxiter': 100}

res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X, y, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of OptimizeResult object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property of the result
theta = res.x

plotDecisionBoundary(plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)
pyplot.show()

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')
