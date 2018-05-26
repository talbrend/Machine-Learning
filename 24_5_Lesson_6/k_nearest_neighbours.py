# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:28:08 2018

@author: Owner
"""
import numpy as np
from sklearn import datasets
import copy

def sort_array_by_column(arr, column):
    l = arr.tolist()
    sorted_list = sorted(l,key=lambda elem: elem[column])
    new_arr = np.array(sorted_list)
    return new_arr
    

def shuffle_array(arr):
    shuffled_arr = copy.deepcopy(arr)
    np.random.shuffle(shuffled_arr)
    return shuffled_arr

def calc_distance(point_1, point_2,nof_columns):
    distance_before_root = 0
    for i in range(nof_columns):
        distance_before_root += np.power(point_1[i]-point_2[i],2)
    distance = np.sqrt(distance_before_root)
    return distance

def meld_arrays(main_array, added_array):
   l = [[added_array[i]] for i in range(added_array.size)] 
   return np.insert(main_array,[main_array[0].size],l,axis=1)

def find_k_nearest_neighbours(arr,point,nof_point_coordinates,k):
    nof_columns = arr.shape[1]
    zeros = np.zeros(arr.shape[0])
    arr_with_distance = meld_arrays(arr, zeros)
    for i in range(arr_with_distance.shape[0]):
        # place distance in the last column of this array
        arr_with_distance[i][nof_columns] = calc_distance(arr_with_distance[i], point,nof_point_coordinates)
    
    #Sort the array by distance which is in column number nof_columns
    sorted_arr_with_distance = sort_array_by_column(arr_with_distance, nof_columns)
    return sorted_arr_with_distance[0:k]


def predict_from_k_nearest_neighbours(arr,point,nof_point_coordinates,k,vote,nof_categories):
    k_nearest_neighbours = find_k_nearest_neighbours(arr,point,nof_point_coordinates,k)
    if vote:
        list_of_occurences_by_label = [0 for i in range(nof_categories)]
    else:
        sum_of_labels = 0
        
    for i in range(k_nearest_neighbours.shape[0]):
        label = k_nearest_neighbours[i][k_nearest_neighbours.shape[1] -2 ]
        if vote:
            list_of_occurences_by_label[int(label)]+= 1
        else:
            sum_of_labels += label
    
    if vote:
        return list_of_occurences_by_label.index(max(list_of_occurences_by_label))
    else:
        return round(sum_of_labels/k)
    
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print("k_nearest_neighbours : " + str(k_nearest_neighbours))
#print("average_label : " + str(average_label))
#print("vote : " + str(vote))
        
def find_precision(training_array, test_array,k):
    nof_samples = training_array.shape[0]
    nof_correct_predictions = 0
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Find Precision according to average: ")
    for i in range(nof_samples):
        point = test_array[i]
        prediction = predict_from_k_nearest_neighbours(training_array,point,4,k,False,3)
        if prediction == point[4]:
            nof_correct_predictions+= 1
    print("Percent of correct predictions: " + str(100*(nof_correct_predictions/nof_samples)) + "\n")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Find Precision according to Voting: ")
    nof_correct_predictions = 0
    for i in range(nof_samples):
        point = test_array[i]
        prediction = predict_from_k_nearest_neighbours(training_array,point,4,k,True,3)
        if prediction == point[4]:
            nof_correct_predictions+= 1
    print("Percent of correct predictions: " + str(100*(nof_correct_predictions/nof_samples)) + "\n")    
           


def main(k,training_data_percentage):
    
    # fetch iris object
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # meld data with target (label)
    data_with_target = meld_arrays(data, target)
    # shuffle the array
    shuffled_data_with_target = shuffle_array(data_with_target)

    # split into training and testing data
    training_array_size =  int(shuffled_data_with_target.shape[0]*training_data_percentage)   
    training_array = shuffled_data_with_target[0:training_array_size]
    test_array = shuffled_data_with_target[training_array_size:shuffled_data_with_target.shape[0]]

    find_precision(training_array, test_array,k)





