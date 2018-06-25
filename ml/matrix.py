import numpy as np

# a=np.matrix('1,2;3,4;5,6')
a=np.array([[1,2],[3,4],[5,6]])
print(a)

#size of the matrix m*n
print(np.shape(a))

#Return row
print(np.shape(a)[0])

#second row of the matrix
print(a[1][:])

#second column of the matrix
print(a[:,1])

#3rd row * 2nd column
print(a[2,1])

b=np.matrix('7,8;9,10')
print(b)

#matrix addition
x=np.matrix('11,12;13,14;15,16')
print(a+x)

#matrix multiplication
c=a.dot(b)
print(c)

#transpose of a matrix
d=a.transpose();
print(d)

#random 3*3 matrix
e=np.random.random((3,3))
print(e)

#random 3*3 matrix of integers
f=np.random.random_integers(0,100,(3,3))
print(f)

#identity matrix 3*3
i=np.eye(3)
print(i)

#inverse of a matrix
g=np.matrix('1,2,3;0,1,-2;1,2,5')
g_inverse=np.linalg.inv(g)
print(g_inverse)

#Vector
v=np.matrix('1;2;3;4')
print(v)

#Sum of vector
print(np.sum(v))

#Sequential vector
ve=np.arange(3)[:,np.newaxis]
print(ve)

#Good to initialize with zero then assign new value
p=np.zeros((10,1),int)
for i in range(10):
    p[i]=2**i

print(p)
print(p[0])