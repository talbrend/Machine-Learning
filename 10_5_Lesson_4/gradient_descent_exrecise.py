# -*- coding: utf-8 -*-
"""
Created on Fri May 11 21:55:22 2018

@author: Owner
"""
import numpy as np
import copy

## h'(theta_j) = 1/M * Sigma (i=1)(theta_0 + theta_1*x + theta_2*x^2 - y(i)) * x(i)(j)

## theta_j = theta_j - alpha*h'(theta_j)

def regression (x,y, dim):
    y_matrix = np.array(y)
    y_matrix = y_matrix.reshape((4,1))
    x_matrix = np.array(x)
    x_transpose = np.transpose(x_matrix)
    x_transpose_x = np.matmul(x_transpose,x_matrix)
    print("x_transpose_x")
    print(x_transpose_x)
    x_transpose_x_inverse = np.linalg.inv(x_transpose_x)
    print("x_transpose_x_inverse")
    print(x_transpose_x_inverse)
    x_transpose_x_inverse_x_transpose = np.matmul(x_transpose_x_inverse,x_transpose)
    h = np.matmul(x_transpose_x_inverse_x_transpose,y_matrix)
    h = h.reshape(dim,)
    return h

def loss (x,y,h):
    loss = 0
    for i in range(len(x)):
        y_ = 0
        for j in range(len(x[i])):
            y_ += h[j]*x[i][j]
        loss += np.power(y[i]-y_,2)
    return loss

def loss_gradient(theta_0, theta_1, theta_2, M, x, y):
    loss = 0
    for i in range(len(x)):
        loss += np.power(theta_0 + theta_1*x[i] + theta_2*x[i]*x[i] - y[i],2)
    return (1/(2*M)) * loss

def get_value_of_theta_coefficient_at_j(j,coefficient):
    if j == 0:
        return 1
    if j == 1:
        return coefficient
    if j == 2:
        return coefficient * coefficient

def gradient_at_theta_j(index_of_theta,theta_0, theta_1, theta_2, M, x, y):
    ret_val = 0
    for i in range(len(x)):
        ret_val += (theta_0 + theta_1*x[i] + theta_2*x[i]*x[i] - y[i]) * get_value_of_theta_coefficient_at_j(index_of_theta,x[i])
    return (1/(M)) * ret_val

def gradient_descent(nof_iterations, theta, alpha,x,y):
    difference_theta = []
    for iteration in range(nof_iterations):
        copy_theta = copy.deepcopy(theta)
        for j in range(len(theta)):
            copy_theta[j] = theta[j] - alpha * gradient_at_theta_j(j,theta[0], theta[1], theta[2], len(x), x, y)
        difference_theta.append ([a_i - b_i for a_i, b_i in zip(theta, copy.deepcopy(copy_theta))])
        theta = copy.deepcopy(copy_theta)
        #print("theta: " + str(theta) + "\n")
        #print("difference_theta: " + str(difference_theta[iteration]) + "\n")
    return theta, difference_theta

def find_best_alpha(nof_iterations, small_alpha, big_alpha, hop,starting_point, x, y):
    i = 0
    direction = -1
    loss = loss_gradient(starting_point[0], starting_point[1], starting_point[2], len(x), x, y)
    alpha = big_alpha
    previous_loss = loss
    while True:
           result = gradient_descent(nof_iterations, starting_point, alpha,x,y)
           final_theta = result[0]
           loss = loss_gradient(final_theta[0], final_theta[1], final_theta[2], len(x), x, y)
           if (loss > previous_loss and i != 0):
               direction *= -1
               hop = hop*0.6
           alpha += direction*hop
           previous_loss = loss
           print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
           print("iteration number " + str(i) + ": alpha = " + str(alpha) + "\n")
           print("loss = " + str(loss) + ", direction: " + str(direction) + " , hop: " + str(hop) + "\n")
           print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
           i+=1
           if i==5000 or hop == 0:
               print("find_best_alpha ended with hop == :" + str(hop) + "\n" )
               break
    return alpha
    

## Linear regression

## 1.a
x = [[31,22],[22,21],[40,37],[26,25]]
y = [2,3,8,12]


h = np.ndarray.tolist(regression(x,y,2))
print ("loss for h: " + str(loss(x,y,h)))
print("h:")
print(h)

## 1.b

x = [[31,22,9],[22,21,1],[40,37,3],[26,25,1]]
y = [2,3,8,12]
h_ = np.ndarray.tolist(regression(x,y,3))
print ("loss for h: " + str(loss(x,y,h_)))

print("h_:")
print(h_)


## Gradient descent

x_gradient = [0,1,2,3]
y_gradient = [1,3,7,13] 

print("Gradient descent exercises:")

## 1.A
alpha = [1,0.1,0.01]
alpha = [0.001]
nof_iterations = 100
starting_point = [2,2,0]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("1.A\n")
loss_at_starting_point = loss_gradient(starting_point[0], starting_point[1], starting_point[2], 4, x_gradient, y_gradient)
print("loss at starting point (2,2,0): " + str(loss_at_starting_point) + "\n")

for i in range(len(alpha)):
    result = gradient_descent(nof_iterations, starting_point, alpha[i],x_gradient,y_gradient)
    final_theta = result[0]
    difference_theta = result[1]
    #print("for alpha == " + str(alpha[i]) + "\nfinal theta is: " + str(final_theta))
    loss_at_end_point = loss_gradient(final_theta[0], final_theta[1], final_theta[2], 4, x_gradient, y_gradient)
    #print("loss is: " + str(loss_at_end_point) + "\n")
    
best_alpha = find_best_alpha(100, 0.01, 1, 0.01,starting_point, x_gradient, y_gradient)
print("best alpha is : " + str(best_alpha) + "\n")
    
           
        


