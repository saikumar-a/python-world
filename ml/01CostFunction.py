import numpy as np

X=np.array([[1,1],[1,2],[1,3]])
print(X)

y=np.array([[1],[2],[3]])
print(y)

theta=np.array([[0],[0]])
# theta=np.array([[0],[1]])
print(theta)

 
def costFunctionJ(X, y, theta):
    m=np.shape(X)[0]
    predictions=X.dot(theta)
    sqrErrors=(predictions-y)**2
    
    J=1/(2*m)*np.sum(sqrErrors)
    
    return J

print(costFunctionJ(X, y, theta))
     
    