import numpy as np
# Import regular expressions to process emails
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
from Ex6.utils import plotData, svmTrain, visualizeBoundaryLinear, linearKernel,\
    visualizeBoundary, svmPredict, getVocabList, PorterStemmer

data = loadmat('ex6data1.mat')
X, y = data['X'], data['y'][:, 0]

# Plot training data
# plotData(X, y)

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 100

# model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
# visualizeBoundaryLinear(X, y, model)
# pyplot.show()


def gaussianKernel(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.
    """
    sim = 0
    # ====================== YOUR CODE HERE ======================
    i_minus_j_sq = np.power(x1 - x2, 2)
    numer = -np.sum(i_minus_j_sq)
    
    denom = 2*sigma*sigma
    sim = np.exp(numer/denom)
    # =============================================================
    return sim

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
      '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))


data = loadmat('ex6data2.mat')
X, y = data['X'], data['y'][:, 0]

# Plot training data
# plotData(X, y)
# pyplot.show()

# SVM Parameters
C = 1
sigma = 0.1

# model= svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# visualizeBoundary(X, y, model)
# pyplot.show()
# print("done!")


data = loadmat('ex6data3.mat')
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# plotData(X, y)
# pyplot.show()


def dataset3Params(X, y, Xval, yval):
    """
    Returns your choice of C and sigma for Part 3 of the exercise 
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel.
    """
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3



    # ====================== YOUR CODE HERE ======================
    start = 0.01
    cs_to_try = 10
    sigmas_to_try = 10
    
    min_err = 10000
    for c in range(1, cs_to_try):
        for s in range(1, sigmas_to_try):
            this_C = start*3*c
            this_s = start*3*s
            model = svmTrain(X, y, this_C, gaussianKernel, args=(this_s,))
            predictions = svmPredict(model, Xval)
            err = np.mean(predictions != yval)
            if err < min_err:
                min_err = err
                C = this_C
                sigma = this_s
                print("New min found!")
                print((this_C, this_s, err))
    # ============================================================
    return C, sigma

# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
# model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
# model = svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# visualizeBoundary(X, y, model)
# pyplot.show()
# print(C, sigma)


def processEmail(email_contents, verbose=True):
    """
    Preprocesses the body of an email and returns a list of indices 
    of the words contained in the email.    
    
    Parameters
    ----------
    email_contents : str
        A string containing one email. 
    
    verbose : bool
        If True, print the resulting email after processing.
    
    Returns
    -------
    word_indices : list
        A list of integers containing the index of each word in the 
        email which is also present in the vocabulary.
    
    Instructions
    ------------
    Fill in this function to add the index of word to word_indices 
    if it is in the vocabulary. At this point of the code, you have 
    a stemmed word from the email in the variable word.
    You should look up word in the vocabulary list (vocabList). 
    If a match exists, you should add the index of the word to the word_indices
    list. Concretely, if word = 'action', then you should
    look up the vocabulary list to find where in vocabList
    'action' appears. For example, if vocabList[18] =
    'action', then, you should add 18 to the word_indices 
    vector (e.g., word_indices.append(18)).
    
    Notes
    -----
    - vocabList[idx] returns a the word with index idx in the vocabulary list.
    
    - vocabList.index(word) return index of word `word` in the vocabulary list.
      (A ValueError exception is raised if the word does not exist.)
    """
    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    
    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
        
    # TOMMY: MODIFIED HERE, CONVERT TO DICT
    v_dict = {}
    for vindex, v in enumerate(vocabList):
        v_dict[v] = vindex
    
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if found
        # ====================== YOUR CODE HERE ======================
        # TOMMY: I only added these two lines along with the above 3 (See note)
        if word in v_dict:
            word_indices.append(v_dict[word])
        # =============================================================

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices

#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

# Extract Features
with open('spamSample1.txt') as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)

#Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)
