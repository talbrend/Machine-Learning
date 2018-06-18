# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:10:56 2018

@author: tb945172
"""

import numpy as np
import pandas as pd
import copy
from scipy.stats import norm

def shuffle_array(arr):
    shuffled_arr = copy.deepcopy(arr)
    np.random.shuffle(shuffled_arr)
    return shuffled_arr

def get_training_and_test(training_data_percentage,mat):
    shuffled_data = shuffle_array(mat)
        # split into training and testing data
    training_array_size =  int(shuffled_data.shape[0]*training_data_percentage)   
    training_array = shuffled_data[0:training_array_size]
    test_array = shuffled_data[training_array_size:shuffled_data.shape[0]]

    return training_array, test_array

# Receive mat and return a matrix with mat[nof_column] == value without last column
def get_matrix_with_column_value(mat, nof_column, value):
    tmp_mat = mat[mat[:,nof_column] == value]
    ret_mat = tmp_mat[:,:-1]
    return ret_mat

def get_array_of_means_for_given_target(mat):
    return np.mean(mat, axis=0)

def get_array_of_stds_for_given_target(mat):
    return np.std(mat, axis=0)

def calc_probability(row_in_mat,mean_mat, std_mat,nof_instances_in_class):
    norm_array = np.empty(mean_mat.shape,dtype=object)
    for i in range(mean_mat.shape[0]):
        norm_array[i] = norm(mean_mat[i],std_mat[i])
    prob_array = np.zeros((row_in_mat.shape[0]+1,))
    prob_array[prob_array.shape[0]-1] = nof_instances_in_class
    for i in range(row_in_mat.shape[0]):
            prob_array[i] = norm_array[i].pdf(row_in_mat[i])
    
    return np.prod(prob_array)

def get_array_of_predictions(test_mat,mean_for_target_zero, mean_for_target_one, std_for_target_zero,std_for_target_one, nof_zeros, nof_ones):
    array_of_predictions = np.zeros((test_mat.shape[0],))
    for i in range(test_mat.shape[0]):
        probability_for_zero = calc_probability(test_mat[i],mean_for_target_zero, std_for_target_zero,nof_zeros)
        probability_for_one = calc_probability(test_mat[i],mean_for_target_one, std_for_target_one,nof_ones)
        array_of_predictions[i] = 0 if probability_for_zero > probability_for_one else 1
    return array_of_predictions
        

def naive_bayes(percentage_of_training_size,mat):
    training_mat, test_mat = get_training_and_test(percentage_of_training_size,mat)
    
    # Get a matrix with rows that have target == 0, target == 1
    training_mat_with_zero_as_target = get_matrix_with_column_value(training_mat, training_mat.shape[1]-1, 0)
    training_mat_with_one_as_target = get_matrix_with_column_value(training_mat, training_mat.shape[1]-1, 1)
    nof_zeros = training_mat_with_zero_as_target.shape[0]
    nof_ones = training_mat_with_one_as_target.shape[0]
    
    # Get the means and stds for target == 0, target == 1
    mean_for_target_zero = get_array_of_means_for_given_target(training_mat_with_zero_as_target)
    mean_for_target_one = get_array_of_means_for_given_target(training_mat_with_one_as_target)
    std_for_target_zero = get_array_of_stds_for_given_target(training_mat_with_zero_as_target)
    std_for_target_one = get_array_of_stds_for_given_target(training_mat_with_one_as_target)
    
    test_mat_without_target_column = test_mat[:,:-1]
    array_of_predictions = get_array_of_predictions(test_mat_without_target_column,mean_for_target_zero, mean_for_target_one, std_for_target_zero,std_for_target_one, nof_zeros, nof_ones)
    
    nof_correct_predictions = 0
    for i in range(test_mat.shape[0]):
        if test_mat[i][test_mat.shape[1]-1] == array_of_predictions[i]:
            nof_correct_predictions += 1
    return (nof_correct_predictions / test_mat.shape[0]) * 100
    

mat = np.genfromtxt("pima-indians-diabetes.csv", delimiter=',')
for i in [x * 0.1 for x in range(1, 10)]:
    percentage_of_correct_predictions = naive_bayes(i,mat)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("percentage of correct predictions for training set with size " + str(i) + " : " + str(percentage_of_correct_predictions))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")




