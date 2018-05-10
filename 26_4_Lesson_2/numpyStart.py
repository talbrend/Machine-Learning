# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:31:25 2018

@author: tb945172
"""

import numpy as np

print ("numpy version:")
print (np.__version__)

a = np.zeros(10)

b = np.zeros(20)

def memory_size(array):
    memory = 0
    shapush = array.shape
    range_x = shapush[0]
    range_y = shapush[1]
    
    for i in range(range_x):
        for j in range(range_y):
            memory += array[i][j].itemsize
    return memory

c = np.array(range(10,50))
d = np.arange(10,50)

e = np.array(range(0,9))
e = np.reshape(e,(3,3))

f= np.eye(3, k=0)
g = np.random.rand(3,3)
l = np.random.rand(10,10)*1000
max_l = l.max()

m = np.random.rand(30)
print(m.mean())
