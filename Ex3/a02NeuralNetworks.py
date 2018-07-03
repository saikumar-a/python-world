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

# randomly permute examples, to be used for visualizing one 
# picture at a time
indices = np.random.permutation(m)


# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
# print(rand_indices)
sel = X[rand_indices, :]
# sel = X[2998, :]

# displayData(sel)


# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary 
weights = loadmat('ex3weights.mat')

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

def sigmoid(z):
     z = np.array(z)
     return 1 / (1 +  np.e**(-z))


def predict(Theta1, Theta2, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    # add the ones, copied from above
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    # print(X.shape)
    # print(Theta1.shape)
    # print(Theta2.shape)
    # (5000, 401)
    # (25, 401)
    # (10, 26)
    # .... based on these dimensions, think we are going to have to do:
    # 1 sig(X * Theta1 ^ T) = 5000*25,   
    # 2 add ones column, giving us 5000*26,
    # 3 do THAT* theta2 ^ T giving us 5000*10, 
    # 4 do our argmax thing
    
    # 1
    z2 = np.matmul(X, Theta1.T)
    a2 = sigmoid(z2)

    # 2
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)
    
    # 3
    z3 = np.matmul(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    # 4
    p = np.argmax(a3, axis=1)

    # =============================================================
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')