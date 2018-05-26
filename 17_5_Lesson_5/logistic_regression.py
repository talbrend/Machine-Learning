# -*- coding: utf-8 -*-
"""
Created on Sun May 20 18:44:14 2018

@author: Owner
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

EPS = 1e-7

def return_x_y_for_line(lower_x,upper_x,number,theta):
    x = np.linspace(lower_x,upper_x, num=number)
    
    y = np.array([(-1*theta[2]-item*theta[0])/theta[1] for item in list(x)])
    return x,y

## 4
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

## 1
def generate_random_points_around_point(x_of_point, y_of_point, sd):
    coordinates_around_x = np.random.normal(x_of_point, sd, 100)
    coordinates_around_y = np.random.normal(y_of_point, sd, 100)
    return coordinates_around_x, coordinates_around_y

## 5

def prediction(theta,x):
    dot_product = np.dot(theta,x)
    if sigmoid(dot_product) > 0.5 :
        return 1
    return 0

## 6
def negative_log_likelihood(y,X,theta, is_circle):
    factor = -1/(X.size)
    sigma_sum = 0
    for i in range(X.shape[0]):
        if is_circle == True :
            dot_product = np.power(X[i][0]-theta[0],2) + np.power(X[i][1]-theta[1],2) - np.power(theta[2],2) 
        else :
            dot_product = np.dot(theta,X[i])
        sigmo = sigmoid(dot_product)
        sigma_sum += y[i] * np.log(sigmo+EPS) + (1-y[i]) * np.log(1-sigmo+EPS)
    return factor * sigma_sum

## 7
def gradient_of_negative_log_likelihood(y,X,theta,j, is_circle):
    sigma_sum = 0
    for i in range(X.shape[0]):
        if is_circle == True :
            inner_derivative = [2*theta[j] - 2*X[i][j], 2*theta[j] - 2*X[i][j],  -2*theta[j]] 
            dot_product = np.power(X[i][0]-theta[0],2) + np.power(X[i][1]-theta[1],2) - X[i][2]*np.power(theta[2],2)
        else :
            inner_derivative = [X[i][j],X[i][j],X[i][j]]
            dot_product = np.dot(theta,X[i])
        sigmo = sigmoid(dot_product)
        term = y[i] - sigmo
        sigma_sum += inner_derivative[j] * term
    return sigma_sum

## 8
def logistic_regression(nof_iterations, theta, alpha,x,y,is_circle):
    difference_theta = []
    for iteration in range(nof_iterations):
        copy_theta = copy.deepcopy(theta)
        for j in range(len(theta)):
            copy_theta[j] = theta[j] + alpha * gradient_of_negative_log_likelihood(y,x,copy_theta,j,is_circle)
        difference_theta.append ([a_i - b_i for a_i, b_i in zip(theta, copy.deepcopy(copy_theta))])
        theta = copy.deepcopy(copy_theta)
        loss = negative_log_likelihood(y,x,copy_theta , is_circle)
        print("theta: " + str(theta) + "\n")
        print("difference_theta: " + str(difference_theta[iteration]) + "\n")
        print("loss is: " + str(loss))
    return theta, difference_theta
    

## 2

point_a = [1,2]
point_b = [3,3]

x_of_point_a, y_of_point_a = generate_random_points_around_point(point_a[0], point_a[1], 0.7);
x_of_point_b, y_of_point_b = generate_random_points_around_point(point_b[0], point_b[1], 0.7);

plt.figure()
plt.scatter(x_of_point_a,y_of_point_a);
plt.scatter(x_of_point_b,y_of_point_b, c="red");

## 3

X = np.array((x_of_point_a,x_of_point_b))
X = X.reshape((x_of_point_a.size + x_of_point_b.size,1))

x_array = np.array((x_of_point_a,x_of_point_b))
x_array = x_array.reshape((x_of_point_a.size + x_of_point_b.size,))
y_array = np.array((y_of_point_a,y_of_point_b))
y_array = y_array.reshape((y_of_point_a.size + y_of_point_b.size,))
ones = [1 for i in range(x_of_point_a.size + x_of_point_b.size)]
data_points = list(zip(x_array,y_array,ones))
X = np.array(data_points)

y = np.array(((np.ones(100),np.zeros(100))))
y = y.reshape(X.shape[0])


# 9
nof_iterations = 100
alpha = [0.01]
starting_point = start =  np.append(np.random.normal(0,0.1,(2)),0)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
loss_at_start_point = negative_log_likelihood(y,X,starting_point, False)
print("Initial loss is: " + str(loss_at_start_point) + "\n")

for i in range(len(alpha)):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result = logistic_regression(nof_iterations, starting_point, alpha[i],X,y, False)
    final_theta = result[0]
    difference_theta = result[1]
    print("Logistic Regrssion for alpha == " + str(alpha[i]) + "\nfinal theta is: " + str(final_theta))
    loss_at_end_point = negative_log_likelihood(y,X,final_theta , False)
    print("loss is: " + str(loss_at_end_point) + "\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    x_of_line, y_of_line = return_x_y_for_line(-3,3,100,final_theta)
    plt.plot(x_of_line, y_of_line)
    
    
## 10
def generate_circle(x_of_center, y_of_center, R,nof_points):
    x_array = np.random.uniform(x_of_center - R,R + x_of_center,(nof_points,))
    factor_array = np.array( [np.random.choice(np.array([-1,1])) for i in range(nof_points)])
    y_array = factor_array * (np.sqrt(R*R - np.power(x_array - x_of_center,2))) + y_of_center
    return x_array, y_array

# Add gaussian noise to array
def add_gaussian_noise_to_array(array):
    
    new_array = np.copy(array)
    
    for point_index in range(array.shape[0]):
       gaussian_noise = np.random.normal(0,0.3)
       new_array[point_index] += gaussian_noise

    return new_array

x_array_1, y_array_1 = generate_circle(3, 2, 5,100)
x_array_1 = add_gaussian_noise_to_array(x_array_1)
y_array_1 = add_gaussian_noise_to_array(y_array_1)
plt.figure()
plt.scatter(x_array_1,y_array_1, c="green");

x_array_2, y_array_2 = generate_circle(3, 2, 9,100)
x_array_2 = add_gaussian_noise_to_array(x_array_2)
y_array_2 = add_gaussian_noise_to_array(y_array_2)
plt.scatter(x_array_2,y_array_2, c="purple");


x_array = np.array((x_array_1,x_array_2))
x_array = x_array.reshape((x_array_1.size + x_array_2.size,))
y_array = np.array((y_array_1,y_array_2))
y_array = y_array.reshape((y_array_1.size + y_array_2.size,))
ones = [1 for i in range(x_array_1.size + x_array_2.size)]
data_points = list(zip(x_array,y_array,ones))
X_ = np.array(data_points)

y_ = np.array(((np.ones(100),np.zeros(100))))
y_ = y.reshape(X_.shape[0])


nof_iterations = 30
alpha = [0.01]
starting_point = [2,1,2]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
loss_at_start_point = negative_log_likelihood(y_,X_,starting_point, True)
print("Initial loss for circle: " + str(loss_at_start_point) + "\n")

for i in range(len(alpha)):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result = logistic_regression(nof_iterations, starting_point, alpha[i],X_,y_, True)
    final_theta = result[0]
    difference_theta = result[1]
    print("Logistic Regrssion for alpha == " + str(alpha[i]) + "\nfinal theta is: " + str(final_theta))
    loss_at_end_point = negative_log_likelihood(y_,X_,final_theta , True)
    print("loss is: " + str(loss_at_end_point) + "\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #x_of_line, y_of_line = return_x_y_for_line(-3,3,100,final_theta)
    #plt.plot(x_of_line, y_of_line)
