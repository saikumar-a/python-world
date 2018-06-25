# Running Notes
1. Most of the programming languages like C, C++, Java use braces { } to define a block of code. Python uses indentation.
```python
for i in range(1,11):
    print(i)
    if i == 5:
        break
```     
2. In Python, we use the hash (#) symbol to start writing a comment.
```python
#This is a comment
#print out Hello
print('Hello')
```
3. For multiline comments use triple quotes, either ''' or """.
```python
"""This is also a
perfect example of
multi-line comments"""
```
4. Docstring is short for documentation string. It is a string that occurs as the first statement in a module, function, class, or method definition. We must write what a function/class does in the docstring. Triple quotes are used while writing docstrings.
```python
def double(num):
    """Function to double the value"""
    return 2*num
```
