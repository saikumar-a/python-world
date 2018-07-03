# Scientific and vector computation for python
import numpy as np
# Plotting library
from matplotlib import pyplot
# Optimization module in scipy
from scipy import optimize
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
from Ex3.utils import displayData

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
# print(rand_indices)
# sel = X[rand_indices, :]
sel = X[2998, :]

displayData(sel)
pyplot.show()

# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

# print(y_t)

def sigmoid(z):
     z = np.array(z)
     return 1 / (1 +  np.e**(-z))
 
def h(X, theta):
    """compute h(x) = sigmoid(X*theta)
       this is the matrix version, this simultaneously multiplies for all feature rows
    """
    theta_t_x = np.matmul(X, theta)
    return sigmoid(theta_t_x)

def lrCostFunction(theta, X, y, lambda_):
    #Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================
    # TOMMY: Copied right out of Ex2, I HAD ALREADY DONE THIS
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

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]');


def oneVsAll(X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================

    
    # ok so there are 5000 samples, and 400 features, and 10 classes.
    # that means
    #  - theta will be 10x401, 1 for the dummy variable.
    #  - X will be 5000*401
    #  - y will be 5000*1
    for c in range(0, num_labels):  # num_label classes

        # Run minimize to obtain the optimal theta. This function will 
        # return a class object where theta is in `res.x` and cost in `res.fun`
        res = optimize.minimize(lrCostFunction,
                                np.zeros(n + 1),
                                (X, y==c, lambda_),  # Extra arguments passed to the objective function and its derivatives, tommy guess: these gets passed to costFunction
                                jac=True,  # If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function
                                method='TNC',
                                options={'maxiter': 400}) 
        all_theta[c] = res.x
            
    # ============================================================
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    # X is 5000*401, theta is 10*401
    # First we will form a matrix 5000*10 = 
    #    [p(row1=0),...,(row1=9),
    #      ....
    #     p(row5000=0),...,(row5000=9)]
    # This is formed by sigmoid(X * theta^T). Suppose we could also do X^T * theta
    # Note. regarding slodes showing h_x = g(theta^T * x), one row of x is 1*401, one row of theta is 1*401, so 
    intermediate = sigmoid(np.matmul(X, np.transpose(all_theta)))
    
    p = np.argmax(intermediate, axis=1)  
    # found via https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    # >>> a = np.arange(6).reshape(2,3)
    # >>> a
    # array([[0, 1, 2],
    #       [3, 4, 5]])
    # >>> np.argmax(a, axis=1)
    # array([2, 2])

    # ============================================================
    return p
# print(X)
pred = predictOneVsAll(all_theta, X)
print(pred[2998])
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))