import os
# Scientific and vector computation for python
import numpy as np
# Plotting library
from matplotlib import pyplot
# Optimization module in scipy
from scipy import optimize
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
from Ex3.utils import displayData
from Ex4.utils import checkNNGradients, predict

data = loadmat('ex4data1.mat')
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
# displayData(sel)

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = loadmat('ex4weights.mat')

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters 
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

def sigmoid(z):
     z = np.array(z)
     return 1 / (1 +  np.e**(-z))

def sigmoidGradient(z):
    sigz = sigmoid(z)
    return np.multiply(sigz, 1-sigz)

def h(mat_or_vec, X, Theta1, Theta2, num_labels):
    if mat_or_vec == "mat":
        return h_mat(X, Theta1, Theta2, num_labels)
    else:
        return h_and_z_vec(X, Theta1, Theta2, num_labels)
    
def h_mat(X, Theta1, Theta2, num_labels):
    """
    # This is modified from exersize 3 and actually computes h(X | theta(s))
    # I REMOVED the adding of the ones since done by caller
    # Also modified the output to return a3 directly, not a prediction
    """
    # note a1 = X, and X is assumed to already have added the column of 1s
    z2 = np.matmul(X, Theta1.T)
    a2 = sigmoid(z2)
    # we are doing a matrix version
    # print(X.shape)
    # print(Theta1.shape)
    # print(Theta2.shape)
    # (5000, 401)
    # (25, 401)
    # (10, 26)
    # .... based on these dimensions, think we are going to have to do:
    # 1 sig(X * Theta1 ^ T) = 5000*25,   
    # 2 add ones column, giving us 5000*26, <- here
    # 3 do THAT* theta2 ^ T giving us 5000*10, 
    # essentially need to add a 1 for *every* a2 vector
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    z3 = np.matmul(a2, Theta2.T)
    a3 = sigmoid(z3)
    return a3

def h_and_z_vec(X, Theta1, Theta2, num_labels):
    """
    # In this implementation,
    1) X is a vector
    2) We return h as well as z2, and a2, needed for backprop
    """
    # note a1 = X, and X is assumed to already have added the column of 1s
    a1 = X
    z2 = np.matmul(X, Theta1.T)
    a2 = sigmoid(z2)
    # X is just a single vector
    # print(X.shape)
    # print(Theta1.shape)
    # print(Theta2.shape)
    # (401,)
    # (25, 401)
    # (10, 26)
    # 1 sig(X * Theta1 ^ T) = 1*25,
    # 2 add one to first row, giving us 1*26, <- here
    # 3 do THAT* theta2 ^ T giving us 1*10,
    # essentially need to add just one 1
    a2 = np.insert(a2, 0, 1)
    z3 = np.matmul(a2, Theta2.T)
    a3 = sigmoid(z3)
    return z2, a1, a2, a3

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
     # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    """
    Compute J
    """
    # K is the number of possible labels.
    
    # Add ones to the X data matrix
    X = np.copy(X)  # dont modify the callers scope!!!
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
        
    # this is 5000x10 and is the vector of predictions for each sample 
    # [
    #   [p(x^1 = 0,..., x^1=num_labels)],
    #   ...
    #   [p(x^m = 0,..., x^m=num_labels)]
    # ]
    hX = h("mat", X, Theta1, Theta2, num_labels)
                      
    # implement the note under Part 2 where we make a matrix
    Y = np.zeros((m, num_labels))
    for index, label in np.ndenumerate(y):
        Y[index][label] = 1
    
    # This is now also a 5000x10 matrix 
    # [
    #   [0,.....y[1],...0],
    #   ...
    #   [0,.....y[m],...0]
    #]
    
    #For the first part of this sum We want  
    #[
    #  [p(x[1])=*Y[1]=0, + ... p(x[1]=num_label)*Y[1]=num_label
    #  ...
    #  [p(x[m]=0)*Y[m]=0, + ... p(x[1]=num_label)*Y[m]=num_label
    #]
    
    # Which is the row wise dot product
    # From https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    part_1_vec = np.einsum('ij,ij->i', np.log(hX), -Y)
    
    #now we do the same for part2
    part_2_vec = np.einsum('ij,ij->i', np.log(1-hX), 1-Y)
    
    # now subtract
    sum_over_k = part_1_vec - part_2_vec
    
    J = (1/m)*np.sum(sum_over_k)
    
    #for regularization we just 
    # - chop off the first column of each theta matrix
    # - square them and 
    # - sum them up
    # implementation wise, we actually subtract off the first colum
    reg_term_1 = np.sum(Theta1**2) - np.sum(Theta1[:,0]**2)
    reg_term_2 = np.sum(Theta2**2) - np.sum(Theta2[:,0]**2)
    regularization = (lambda_/(2*m)) * (reg_term_1 + reg_term_2)
    
    J = J + regularization
    
    """
    Backprop
    """
    # The settig of the intial deltas isn't in the pDF or in this notebook,
    # I found it on page 8 of the slides
    # since there is only a delta2 and a delta3, and the forumla uses delta l+1, 
    # I assume we only need Delta1 and Delta2
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    for i in range(m):
        #step 1: get the a3a2 vectors
        z2, a1, a2, a3 = h("vec", np.transpose(X[i]), Theta1, Theta2, num_labels)
        
        #step 2: compute the error terms at the output nodes
        delta3 = a3 - Y[i]
        
        #step 3: compute delta2s
        # PITFALL!!!!
        # Theta2 includes the bias column, but z2 does NOT. 
        # Originally trying to compute theta2^T * delta3 .* sigz2 was failing because
        # theta2^T is a 26x10 matrix, so theta2^Tdelta3 is a 1x26 vector,
        # but z2 and sigz2 are only a 25 column vector.
        # After wracking my brain, I found this note in the forums:
        #Q5) Backpropagation - Step 3 on page 9 of ex4.pdf doesn't work - the dimensions don't match.

        #The instructions for Step 3 and Step 4 are incomplete.
        # There are two ways to implement this:

        #a) In Step 3, if you compute the sigmoid gradient of z2, then you must ignore the first column of Theta2 when you compute \delta^{(2)}
        #or
        #b) In Step 3, if you compute the sigmoid gradient of a2, then you must pay attention to the note about removing the first column of \delta^{(2)} 
         #when you compute \Delta^{(1)} 
            
        # Taking the "a" route
        # going to drop the first column of Theta2 
        Theta2copy = np.copy(Theta2)
        Theta2_unbiased = np.delete(Theta2, 0, axis=1)
    
        sigz2 = sigmoidGradient(z2)
        theta2delta3 = np.matmul(Theta2_unbiased.T, delta3)
        delta2 = np.multiply(theta2delta3, sigz2)
        
        # step 4, accumulate the gradients
        # assert np.array_equal(np.outer(delta3, a2.T), np.outer(delta3, a2))
        # Ran into issue where I couldn't use matmul on delta3 and a2.T, it didn't recognize 
        # the inner dimension as 1
        # https://stackoverflow.com/questions/28578302/how-to-multiply-two-vector-and-get-a-matrix
        Delta2 = Delta2 + np.outer(delta3, a2.T)
        Delta1 = Delta1 + np.outer(delta2, a1.T)
    
    # step 4
    # For the term regularization, we are going to make copies of Theta1 and Theta2, but
    # zero out the first columns

    
    Theta1_zeroed = np.copy(Theta1)
    Theta2_zeroed = np.copy(Theta2)
    
    Theta1_zeroed[:,0] = np.zeros(Theta1_zeroed.shape[0])
    Theta2_zeroed[:,0] = np.zeros(Theta2_zeroed.shape[0])
    
    Theta1_grad = ((1/m) * Delta1) + ((lambda_/m) * Theta1_zeroed)
    Theta2_grad = ((1/m) * Delta2) + ((lambda_/m) * Theta2_zeroed)

        
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

lambda_ = 0
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_)
print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
print('The cost should be about                   : 0.287629.')

# Weight regularization parameter (we set this to 1 here).
lambda_ = 1
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %.6f' % J)
print('This value should be about                 : 0.383770.')

z = np.array([-1, -0.5, 0, 0.5, 1])
g = sigmoidGradient(z)
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print(g)

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

print(initial_nn_params)

checkNNGradients(nnCostFunction)

print("TOMMY NOTE! These matched exactly before adding the regularization term on the thetas")

#  Check gradients by running checkNNGradients
lambda_ = 3
checkNNGradients(nnCostFunction, lambda_)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 100}

#  You should also try different values of lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

print("Working....")
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

displayData(Theta1[:, 1:])

data = loadmat('ex4data1.mat')
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size
