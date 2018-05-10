# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:41:08 2018

@author: Owner
"""

import numpy as np
import random
from numpy.linalg import inv
import math

## 1)

# Get a vector with 
# dim: dimension of vector
# max_range: upper limit for the randomized interval
# return: the vector
def get_random_numbers_vector(dim, max_range):
    return max_range*np.random.rand(dim)
   
# Get a vector with random integers    
# dim: dimension of vector
# min_range :  lower limit for the randomized interval
# max_range: upper limit for the randomized interval
# return: the vector    
def get_random_int_numbers_vector(dim,min_range, max_range):
    return np.random.randint(min_range,max_range,size=dim)

# Get a vector with random integers that are multiples of multiple   
# dim: dimension of vector
# min_range :  lower limit for the randomized interval
# max_range: upper limit for the randomized interval
# return: the vector    
def get_random_vector_with_multiples_of_some_number(dim,min_range, max_range,multiple):
    return multiple*get_random_int_numbers_vector(dim,min_range, max_range)

# Return the nth fibonacci number
def fib(n):
    if n == 0 or n == 1 :
        return n
    else:
        return fib(n-1) + fib(n-2)
        
 # Return a random (0-9) fibonacci number   
def get_random_fib_number():
    random_fib_number = random.randint(0,9)
    return fib(random_fib_number)

## 2)
    
# Get an array of random points on a line
# nof_points : the number of points in the returned vector
# elevation: elevation of the function, i.e. b in ax+b
# slope: slope of the functionm i.e. a in ax+b
# return: array_of_points
def get_array_of_points_on_line(nof_points, slope, elevation):
    # Get a vector with random float values, up to max_range
    # Will hold the x values
    max_range = 100
    random_x_values = get_random_numbers_vector(nof_points, max_range)
    array_of_points = []
    
    # Go over all randomized x values, assign a y value
    # according to slope and elevation
    for point_index in range(nof_points):
        x_value = random_x_values[point_index]
        point = np.array([x_value, slope * x_value + elevation])
        array_of_points.append(point)
        
    return np.asarray(array_of_points)

# Add gaussian noise to array
def add_gaussian_noise_to_array(array):
    
    new_array = np.copy(array)
    
    for point_index in range(array.shape[0]):
       gaussian_noise = np.random.normal(0,1)
       new_array[point_index][1] += gaussian_noise

    return new_array

# Get an array of random points on a parabola
# nof_points : the number of points in the returned vector
# a: a in ax^2 + bx + c
# b: a in ax^2 + bx + c
# c: a in ax^2 + bx + c
# return: array_of_points
def get_array_of_points_on_parabola(nof_points, a, b, c):
    # Get a vector with random float values, up to max_range
    # Will hold the x values
    max_range = 100
    random_x_values = get_random_numbers_vector(nof_points, max_range)
    array_of_points = []
    
    # Go over all randomized x values, assign a y value
    # according to slope and elevation
    for point_index in range(nof_points):
        x_value = random_x_values[point_index]
        point = np.array([x_value, a*x_value*x_value + b*x_value + c])
        array_of_points.append(point)
        
    return np.asarray(array_of_points)
       

first_array = get_array_of_points_on_line(10, 0, random.randint(0,9))

normalized_first_array = add_gaussian_noise_to_array(first_array)

second_array = get_array_of_points_on_line(10, random.randint(0,9), random.randint(0,9))

normalized_second_array = add_gaussian_noise_to_array(second_array)

third_array = get_array_of_points_on_parabola(20, 1, 2, 1)



## 2)

mat_1 = []
mat_2 = []
for i in range(4):
    mat_1.append (get_random_int_numbers_vector(4,0, 10))
    mat_2.append (get_random_int_numbers_vector(4,0, 10))
    
mat_1 = np.asarray(mat_1)
mat_2 = np.asarray(mat_2)
mat_3 = np.matmul(mat_1, mat_2)
transposed_mat_3 = np.transpose(mat_3)
inversed_mat_3 = inv(mat_3)


## 8)

x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]

y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]

# Define : Y1 = ln(y1) - ln(y2) for consecutive y1,y2 in y
# Define : A1 = x2^2 - x1^2 for consecutive x1,x2 in x
# Define: B1 = x2-x1 for consecutive x1,x2 in x
Y = []
A = []
B = []

for i in range(0,len(x),2):
    A.append(x[i+1]*x[i+1]  - x[i]*x[i])
    B.append(x[i+1] - x[i])
    
for i in range(0,len(y),2):
    Y.append(np.log(y[i+1]) - np.log(y[i]))
    
    l1 = []
    l2 = []
    
while True:
    index_1 = random.randint(0,9)
    index_2 = random.randint(0,9)
    if (index_1 == index_2):
        continue
    if (A[index_1] / B[index_1] == A[index_2] / B[index_2]):
        continue
    l1 = [A[index_1],B[index_1]]
    l2 = [A[index_2],B[index_2]]
    matrix = np.array([l1,l2])
    
    break;
    
vec = np.array([Y[index_1], Y[index_2]])
vec = vec.reshape((2,1))

matrix_inverse = inv(matrix)

solution_vector = np.matmul(matrix_inverse,vec)

b = solution_vector[0][0]
c = solution_vector[1][0]

print("index_1: " + str(index_1))
print("index_2: " + str(index_2))
print("solution_vector: " + str(solution_vector))
    
a = []
for i in range(len(x)):
    a.append(y[i] / (math.exp(b*math.pow(x[i],2))) + c*x[i])

print("a: " + str(a))
