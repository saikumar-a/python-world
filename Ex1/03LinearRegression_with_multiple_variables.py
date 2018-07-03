import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

data=np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# print(X)
# print(y)
# print(m)

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
    
def  featureNormalize(X):
     # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    cols = X_norm.shape[1]  # number of cols
    
    for c in range(0, cols):
        col = X_norm[:,c]
        sigma[c] = np.std(col)
        mu[c] = np.mean(col)
    
    rows = X_norm.shape[0]  # number of rows
    for row in range(0, rows):
        for col in range(0, cols):
            X_norm[row][col] = (X_norm[row][col] - mu[col]) / sigma[col]
  
    # ================================================================
    return X_norm, mu, sigma

# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
# print(X)

def computeCostMulti(X, y, theta):
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    term = np.dot(X, theta) - y
    t_cost = np.dot(np.transpose(term), term)
    J = 1/(2*m) * t_cost
    
    # ==================================================================
    return J

print(computeCostMulti)


def computeCost(X, y, theta):
    m=y.size
    J = 0
    
    diffs = [np.dot(np.transpose(theta), X[i]) - y[i] for i in range(0,m)]
    diffs_sq = np.power(diffs, 2)
    J = 1/(2*m) * np.sum(diffs_sq)
    
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        new_theta = np.zeros(theta.size)
        for t in range(0, theta.size):
            theta_diff = [(np.dot(np.transpose(theta), X[i]) - y[i])*X[i][t] for i in range(0,m)]
            new_theta[t] = theta[t] - alpha * 1/m * np.sum(theta_diff)
        theta = new_theta

        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history


# print(gradientDescentMulti)

# Choose some alpha value - change this
alpha = 0.3
num_iters = 50

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# print(theta,J_history)

# Plot the convergence graph
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
# plt.show()


# Estimate the price of a 1650 sq-ft, 3 br house
# ======================= YOUR CODE HERE ===========================
# Recall that the first column of X is all-ones. 
# Thus, it does not need to be normalized.

x = np.matrix([1, 1650, 3])
x_normalized = [1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]]
price = np.dot(np.transpose(theta), x_normalized)   # You should change this

# ===================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))



data=np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    
    # ===================== YOUR CODE HERE ============================
    x_t = np.transpose(X)
    x_t_x = np.dot(x_t, X)
    x_t_x_inv = np.linalg.inv(x_t_x)
    
    x_t_y = np.dot(x_t, y)
    
    theta = np.dot(x_t_x_inv, x_t_y)

    # =================================================================
    return theta

# Calculate the parameters from the normal equation
theta = normalEqn(X, y);

# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta)));

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

price = np.dot(np.transpose(theta), [1, 1650, 3])  # don't need to normalize here

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))

# TOMMY ADDED

data=np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, :2]
y = data[:, 2]
# print(X)

regr = LinearRegression()
regr.fit(X,y)
price=regr.predict(np.array([[1650,3]]))
print('Predicted price of a 1650 sq-ft, 3 br house (using Scikit-learn): ${:.0f}'.format(price[0]))
