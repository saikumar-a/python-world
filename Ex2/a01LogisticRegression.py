# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize
from Ex2 import utils
from Ex2.utils import plotData

from sklearn.linear_model import LogisticRegression

data=np.loadtxt('ex2data1.txt',delimiter=',')
X, y = data[:, 0:2], data[:, 2]

    
plotData(X,y)
# add axes labels
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
# pyplot.show()


def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    return 1 / (1 +  np.e**(-z))

# Test the implementation of sigmoid function here
z = 0
g = sigmoid(z)

print('g(', z, ') = ', g)


# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# TOMMY: defining this here because I will use it in multiple places below
def h(X, theta):
    """compute h(x) = sigmoid(X*theta)
       this is the matrix version, this simultaneously multiplies for all feature rows
    """
    theta_t_x = np.matmul(X, theta)
    return sigmoid(theta_t_x)

def costFunction(theta, X, y):
     # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # used h above
    # COMPUTE J
    # we flip y and X here insetead of transposing y
    J_inner = np.matmul(np.log(h(X, theta)), -y) - np.matmul(np.log(1-h(X, theta)), (1-y))
    J = (1/m) * np.sum(J_inner) 
    
    # COMPUTE GRADIENT
    # remember x_j^(i) j is the feature index (column in X), (i) is the sample index (row in X)
    # there is a theta for each feature, but becayse of theta_0, recall we added a simple "1" to the feature vector
    # so the size of X columnwise == number of thetas == number of features +1
    # so to update the gradient vector all at once, we want to mulitiply (h(X, theta) - y) times the tth column of X
    # to get the vector of gradients, this is just the matrix multiplication of h(X, theta) - y) and X
    grad = (1/m) * np.matmul((h(X, theta) - y), X) 
    
    # =============================================================
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]\n')



# set options for optimize.minimize
options= {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is 
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),  # Extra arguments passed to the objective function and its derivatives, tommy guess: these gets passed to costFunction
                        jac=True,  # If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]\n')

# Plot Boundary
utils.plotDecisionBoundary(plotData, theta, X, y)
# pyplot.show()


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


#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')

# Using Scikit - learn
data=np.loadtxt('ex2data1.txt',delimiter=',')
X, y = data[:, 0:2], data[:, 2]

logisticRegr = LogisticRegression()

logisticRegr.fit(X, y)
prob=logisticRegr._predict_proba_lr(np.array([[45,85]]))
print('\nScikit-learn For a student with scores 45 and 85,'
    'we predict an admission probability of {:.3f}'.format(prob[0][1]))
