# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print('This is a demo code')

# This is function
def foo(x):
    print('x is : ' + str(x))
    
foo(5)
foo('foofoo')


# read from console:
val = input("Please provide a value")

foo(val)

# For statement, range returns values from 0,1,2,3,4 (it is a generator, more on that later)
for i in range(5):
    print(2*i)
    
names = ['Alice','Bob','Eve']

# Another for, this time on list
for name in names:
    # If statement:
    if 'a' in name:
        print(name)    
    # Why didn't we print Alice?
    
    if 'a' in name.lower():
        print(name)
    # lower() is a function, which is called on name, as name is a string class, 
    # the value of name is passed to the function, more on classes later.


# Dictionary - pairs of keys and values for easy access by keys
# creation:
age_dictionary = {'Alice':22,'Bob':27,'Eve':25}
print(age_dictionary['Bob'])
# Error:
print(age_dictionary['Martin'])

# setting value
age_dictionary['Martin'] = 17
print(age_dictionary['Martin'])

# overwritting value:
age_dictionary['Martin'] = 22
print(age_dictionary['Martin'])

    
    

    

