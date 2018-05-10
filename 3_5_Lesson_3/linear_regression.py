# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:30:27 2018

@author: Eli
"""

import numpy as np
from matplotlib import pyplot as plt

random_ints = np.random.randint(0,10,10)
ranom_floats = np.random.rand(10)
random_3_multipliers = np.random.randint(0,10,10) * 3

print('Random integers: ', random_ints)
print('Random floats: ', ranom_floats)
print('Random 3 multipliers: ', random_3_multipliers)

def fibo(n):
    if n==1 or n==0:
        return 1
    return fibo(n-1)+fibo(n-2)

# This is not an efficient implementation!!!
ten_fibo = []
for i in range(10):
    ten_fibo.append(fibo(i))

fibo_index=np.random.randint(0,10)
random_fibo = ten_fibo[fibo_index]
print('Random fibonachi at location {} is: {}'.format(fibo_index, random_fibo))

m_1 = 2
x = ranom_floats
first_array = m_1*x
gaussian_noise = np.random.normal(0,0.01,10)
first_array+=gaussian_noise

m_2=3
n_2 = -1
second_array = m_2*x + n_2
second_array+=gaussian_noise

l_3=2
m_3=1
n_3=2
third_array = l_3*(x**2) + m_3*x + n_3
third_array+=gaussian_noise

mat1 = np.random.rand(4,4)
mat2 = np.random.rand(4,4)
mult= np.dot(mat1,mat2)
print('Matrix multiplication result ', mult)
print('Matrix transpoe ', mult.transpose())
print('Matrix inverse ', np.linalg.inv(mult))


x=x.reshape(10,1)

def regression(x,y):
    assert len(x.shape)==2
    return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)
    
print('first_array results :', regression(x,first_array))
ones_vec = np.ones((10,1))
x2=np.hstack((ones_vec,x))
reg2_res=regression(x2,second_array)
print('second_array results :', reg2_res)

plt.figure()
plt.scatter(x,second_array)
new_x= np.linspace(0,1,11)
plt.plot(new_x, new_x*reg2_res[1]+reg2_res[0])

x3=np.hstack((ones_vec,x, x**2))
reg3_res=regression(x3,third_array)
print('third_array results :', reg3_res)

plt.figure()
plt.scatter(x,third_array)
plt.plot(new_x, (new_x**2)*reg3_res[2]+new_x*reg3_res[1]+reg3_res[0])



x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]
x=(np.array(x)).reshape(len(x),1)
y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]
y=np.array(y)
y_new=np.log(y)

x4=np.hstack((np.ones((20,1)),x, x**2))
reg_res = regression(x4,y_new)
print(np.exp(reg_res[0]),reg_res[1],reg_res[2])