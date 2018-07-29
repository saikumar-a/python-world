import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
from Ex5.utils import trainLinearReg, featureNormalize, plotFit

# Load from ex5data1.mat, where all variables will be store in a dictionary
data = loadmat('ex5data1.mat')

# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector
X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

# m = Number of examples
m = y.size

# Plot training data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)');
# pyplot.show()

def linearRegCostFunction(X, y, theta, lambda_=0.0):
    # Initialize some useful values
    m = y.size # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    diffs = np.matmul(X, theta) - y
    diffs_sq = np.power(diffs, 2)
    J_first_term = 1/(2*m) * np.sum(diffs_sq)
    thetas_squared = np.power(theta, 2)
    sum_thetas_squared_without_0 = np.sum(thetas_squared) - thetas_squared[0]
    J_second_term = lambda_/(2*m)*sum_thetas_squared_without_0
    J = J_first_term + J_second_term
    
    # X^T has the rows of X as the features, e.g., X^T row 1 is feature 0 of all rows 
    # Left multiplying X^T by diffs should give us the gradient equation without regularization
    # Note this required a lot of whiteboarding so if this code isn't obvious later, whiteboard this!
    grad_without_reg = (1/m) * (np.matmul(np.transpose(X), diffs))
    
    # next we compute the reg vector
    reg_vec = (lambda_/m)*theta
    # no reg on first term
    reg_vec[0] = 0
    grad = grad_without_reg + reg_vec
    #BOOM! Vectorized.
    
    # ============================================================
    return J, grad

theta = np.array([1, 1])
J, _ = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Cost at theta = [1, 1]:\t   %f ' % J)
print('This value should be about 303.993192)\n')

theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
print(' (this value should be about [-15.303016, 598.250744])\n')

# add a columns of ones for the y-intercept
X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = trainLinearReg(linearRegCostFunction, X_aug, y, lambda_=0)

#  Plot fit over the data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.plot(X, np.dot(X_aug, theta), '--', lw=2);
# pyplot.show()

def learningCurve(X, y, Xval, yval, lambda_=0):
     # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    for i in range(1, m+1):
        X_i = X[:i, :]
        y_i = y[:i]
        # use the line from the above
        theta_i = trainLinearReg(linearRegCostFunction, X_i, y_i, lambda_=lambda_)
        
        #use their hint to use our original function with a lambda of 0 to compute J_train
        J_train_i = linearRegCostFunction(X_i, y_i, theta_i, lambda_=0.0)[0]
        error_train[i-1] = J_train_i  # note the i-1
        
        J_cross_eval_i = linearRegCostFunction(Xval, yval, theta_i, lambda_=0.0)[0]
        error_val[i-1] = J_cross_eval_i  # note the i-1
        
    # =============================================================
    return error_train, error_val

X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

# pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
# pyplot.title('Learning curve for linear regression')
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 150])
# pyplot.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
    
    

def polyFeatures(X, p):
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))
    # ====================== YOUR CODE HERE ======================
    for r in range(0, X.shape[0]):
        for c in range(0, p):
            X_poly[r][c] = np.power(X[r], c+1)
    # ============================================================
    return X_poly

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

print('Normalized Training Example 1:')
print(X_poly[0, :])


lambda_ = 100
theta = trainLinearReg(linearRegCostFunction, X_poly, y,
                             lambda_=lambda_, maxiter=55)

# Plot training data and fit
# pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
# 
# plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)
# 
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
# pyplot.ylim([-20, 50])
# 
# pyplot.figure()
# error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
# pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)
# 
# pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 100])
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.show()

print('Polynomial Regression (lambda = %f)\n' % lambda_)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
    
def validationCurve(X, y, Xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    # ====================== YOUR CODE HERE ======================
    for lindex, l in enumerate(lambda_vec):
        # use the line from the above
        theta_l = trainLinearReg(linearRegCostFunction, X, y, lambda_=l)
        
        #use their hint to use our original function with a lambda of 0 to compute J_train
        J_train_i = linearRegCostFunction(X, y, theta_l, lambda_=0.0)[0]
        error_train[lindex] = J_train_i  # note the i-1
        
        J_cross_eval_i = linearRegCostFunction(Xval, yval, theta_l, lambda_=0.0)[0]
        error_val[lindex] = J_cross_eval_i  # note the i-1
        

    # ============================================================
    return lambda_vec, error_train, error_val

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('lambda')
pyplot.ylabel('Error')
pyplot.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_test, ytest)

print('lambda\t\tTrain Error\tTest Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))