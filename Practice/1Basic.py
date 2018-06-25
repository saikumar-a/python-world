# from https://www.learnpython.org
# Basic and Variable types

print("HelloWorld")

x=1
if x==1:
    print('x is 1')
    
myInt=7
print(myInt)

myFloat=7.0
print(myFloat)
myFloat=float(7)
print(myFloat)

myString='hello'
print(myString)
myString="world"
print(myString)

myString = "Don't worry about apostrophes"
print(myString)

one = 1
two = 2
three = one + two
print(three)

hello = "hello"
world = "world"
helloworld = hello + " " + world
print(helloworld)

a,b=4,5
print(a)

one = 1
two = 2
hello = "hello"

print(one + two + hello)

# change this code
mystring = "hello"
myfloat = 10.0
myint = 20

# testing code
if mystring == "hello":
    print("String: %s" % mystring)
if isinstance(myfloat, float) and myfloat == 10.0:
    print("Float: %f" % myfloat)
if isinstance(myint, int) and myint == 20:
    print("Integer: %d" % myint)