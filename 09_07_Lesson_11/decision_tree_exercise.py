# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:12:40 2018

@author: tb945172
"""

import numpy as np
import copy
from datetime import datetime
from datetime import date
import time
import timeit

EPS = 0.000001

class Node:
    
    node_id = 0
    node_dict = {}
    
    def __init__(self,data_set):
        self.data_set = data_set
        self.right = None
        self.left = None
        self.father = None
        self.feature_index = -1
        self.split_position = -1
        self.label = 0
        self.depth = 0
        self.id = 0
        self.set_id()
        self.available_feature_indices = []
        
    def set_id(self):
        self.id = Node.node_id
        Node.node_id = Node.node_id + 1
        self.add_self_to_node_dict()
        
    def add_self_to_node_dict(self):
        Node.node_dict[self.id] = self   
        
    def print_node(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("label: " + str(self.label))
        print("data_set: " + str(self.data_set))
        print("feature_vector: " + str(self.data_set[:,1]))
        print("feature_index: " + str(self.feature_index))
        print("split_position: " + str(self.split_position))
        print("depth: " + str(self.depth))
        print("id: " + str(self.id))
        print("<_________________________________________>")

def shuffle_array(arr):
    shuffled_arr = copy.deepcopy(arr)
    np.random.shuffle(shuffled_arr)
    return shuffled_arr
        
def is_data_set_empty(data_set):
    if data_set.shape[0] == 0 or data_set.shape[1] == 0:
        return True
    return False

def shuffle_array(arr):
    shuffled_arr = copy.deepcopy(arr)
    np.random.shuffle(shuffled_arr)
    return shuffled_arr


def meld_arrays(main_array, added_array):
   l = [[added_array[i]] for i in range(added_array.size)] 
   return np.insert(main_array,[main_array[0].size],l,axis=1)

def get_nof_labels_in_dict(dictionary):
    nof_labels = 0
    for val in dictionary.values():
        nof_labels += val
    return nof_labels

def calculate_gini_for_dict(dictionary):
    nof_labels = get_nof_labels_in_dict(dictionary)
    if nof_labels == 0:
        return 0
    sum_of_squares = 0
    
    for val in dictionary.values():
        prob = val / nof_labels
        sum_of_squares += prob*prob
    return (1 - sum_of_squares)
    
    
def calculate_gini_for_feature_value(left_label_dict, right_label_dict):
    nof_left_labels = get_nof_labels_in_dict(left_label_dict)
    nof_right_labels = get_nof_labels_in_dict(right_label_dict)
    
    portion_of_left = 0
    portion_of_right = 0;
    if nof_left_labels != 0:
        portion_of_left = nof_left_labels / (nof_left_labels + nof_right_labels)
    if nof_right_labels != 0:
       portion_of_right = nof_right_labels / (nof_left_labels + nof_right_labels) 
    
    gini_for_left = calculate_gini_for_dict(left_label_dict)
    gini_for_right = calculate_gini_for_dict(right_label_dict)
    
    return portion_of_left*gini_for_left + portion_of_right*gini_for_right
    
    

def init_left_and_right_label_count(list_of_labels):
    left = {}
    right = {}
    for label in list_of_labels:
        frequency = right.get(label)
        if frequency == None:
            right[label] = 1
            continue
        else:
            right[label] = frequency + 1
    left = {key:0 for key in right.keys()}
    return left , right

def delete_column(data_set,column):
    return np.delete(data_set,column,1)

def get_column(data_set,column):
    return data_set[:,column].tolist()
#def kaki(list_of_feature_values):
    
def get_dict_from_lists(list1, list2):
    return dict(zip(list1, list2))

def get_gini_for_feature(feature_values, labels):
    sorted_feature_values = sorted(feature_values)
    dict_of_feature_values_and_labels = get_dict_from_lists(feature_values, labels)
    left_label_dict , right_label_dict  = init_left_and_right_label_count(labels)
    split_position = sorted_feature_values[0] - EPS
    next_val = 0
    min_gini = 1.1
    gini = min_gini
    gini = calculate_gini_for_feature_value(left_label_dict, right_label_dict)
    if gini < min_gini:
        min_gini = gini
    
    for i in range(len(sorted_feature_values)):
        val = sorted_feature_values[i]
        label = dict_of_feature_values_and_labels[val]
        frequency_of_label_in_left = left_label_dict[label]
        frequency_of_label_in_right = right_label_dict[label]
        left_label_dict[label] = frequency_of_label_in_left + 1
        right_label_dict[label] = frequency_of_label_in_right - 1
        
        if i == len(sorted_feature_values) -1:
            next_label = 0
            next_val = sorted_feature_values[i] + EPS
        else :
            next_val = sorted_feature_values[i+1]
            next_label = dict_of_feature_values_and_labels[next_val]
            
        if next_label != label:
            gini = calculate_gini_for_feature_value(left_label_dict, right_label_dict)
            if gini < min_gini:
                min_gini = gini
                split_position = (val + next_val) / 2

    return min_gini, split_position


def split_data(data_set, indx, val):
    mat_bigger = data_set[data_set[:,indx] > val]
    mat_smaller = data_set[data_set[:,indx] <= val]
    returned_list = [mat_bigger, mat_smaller, indx , val]
    return returned_list

def split_data_by_minimal_gini(root, available_feature_indices):
    data_set = root.data_set
    nof_columns = data_set.shape[1]
    labels = data_set[:,1].tolist()
    min_gini = 1.1
    final_split_position = 0
    feature_index = 0
    for column in range(nof_columns - 1):
        if available_feature_indices[column] == False:
            continue
        feature_values = get_column(data_set,column)
        gini, split_position = get_gini_for_feature(feature_values, labels)
        if gini < min_gini:
            min_gini = gini
            final_split_position = split_position
            feature_index = column
        
    returned_list = split_data(data_set, feature_index, final_split_position)
    returned_list.append(gini)
    return  returned_list

def test_get_gini_for_feature(feature_values, labels):
    indx = 1
    sorted_feature_values = sorted(feature_values)
    dict_of_feature_values_and_labels = get_dict_from_lists(feature_values, labels)
    left_label_dict , right_label_dict  = init_left_and_right_label_count(labels)
    EPS = 0.000001
    split_position = sorted_feature_values[0] - EPS
    next_val = 0
    min_gini = 1.1
    gini = min_gini
    gini = calculate_gini_for_feature_value(left_label_dict, right_label_dict)
    if gini < min_gini:
        min_gini = gini
    
    for i in range(len(sorted_feature_values)):
        val = sorted_feature_values[i]
        label = dict_of_feature_values_and_labels[val]
        frequency_of_label_in_left = left_label_dict[label]
        frequency_of_label_in_right = right_label_dict[label]
        left_label_dict[label] = frequency_of_label_in_left + 1
        right_label_dict[label] = frequency_of_label_in_right - 1
        
        if i == len(sorted_feature_values) -1:
            next_label = 0
            next_val = sorted_feature_values[i] + EPS
        else :
            next_val = sorted_feature_values[i+1]
            next_label = dict_of_feature_values_and_labels[next_val]
            
        if next_label != label:
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            #print("Trial number " + str(indx))
            #print("left_label_dict " + str(left_label_dict))
            #print("right_label_dict " + str(right_label_dict))
            #print("index of feature " + str(i))
            gini = calculate_gini_for_feature_value(left_label_dict, right_label_dict)
            #print("gini" + str(gini))
            indx = indx + 1
            if gini < min_gini:
                min_gini = gini
                split_position = (val + next_val) / 2

    return min_gini, split_position

def test_split_data_by_minimal_gini(root, available_feature_indices):
    data_set = root.data_set
    nof_columns = data_set.shape[1]
    labels = data_set[:,1].tolist()
    min_gini = 1.1
    final_split_position = 0
    feature_index = 0
    for column in range(nof_columns - 1):
        if available_feature_indices[column] == False:
            continue
        feature_values = get_column(data_set,column)
        gini, split_position = get_gini_for_feature(feature_values, labels)
        if gini < min_gini:
            min_gini = gini
            final_split_position = split_position
            feature_index = column
        
    returned_list = split_data(data_set, feature_index, final_split_position)
    returned_list.append(gini)
    return  returned_list

def list_to_freq_dict(lst):
    freq = [lst.count(p) for p in lst]
    return dict(zip(freq,lst))

def majority_vote(labels):
    dict_of_freq = list_to_freq_dict(labels)
    return dict_of_freq[max(list(dict_of_freq.keys()))]
    
def train_tree(root, depth_limit, available_feature_indices):
    #nof_columns = root.data_set.shape[1]
    labels = root.data_set[:,1].tolist()
    first_label = labels[0]
    
    if all(label == first_label for label in labels):
        root.label = first_label
        return
    if root.data_set.shape[0] == 1:
        root.label = root.data_set[0][1]
        return
    elif root.depth == depth_limit:
        root.label = majority_vote(labels)
        return

    root.available_feature_indices = copy.deepcopy(available_feature_indices)
    ret = split_data_by_minimal_gini(root, available_feature_indices)
    right_data_set = ret[0]
    left_data_set = ret[1]
    feature_index = ret[2]
    split_position = ret[3]
    gini = ret[4]
    
    root.feature_index = feature_index
    root.split_position = split_position
    
    root.left = Node(left_data_set)
    root.left.depth = root.depth + 1
    root.left.father = root

    root.right = Node(right_data_set)
    root.right.depth = root.depth + 1
    root.right.father = root
    
    new_available_feature_indices = copy.deepcopy(available_feature_indices)
    new_available_feature_indices[feature_index] = False
    
    if is_data_set_empty(left_data_set) == False:
        train_tree(root.left,depth_limit,new_available_feature_indices)
    if is_data_set_empty(right_data_set) == False:
        train_tree(root.right,depth_limit,new_available_feature_indices)
    
    return

def get_label(root, instance):
    if root.label == 0:
        if instance[root.feature_index] > root.split_position:
            if root.right == None:
                #print("Father of none node:")
                root.print_node()
                #print("Father of father of none node:")
                root.father.print_node()
            return get_label(root.right, instance)
        else:
            return get_label(root.left, instance)
    return root.label

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
           
def render_list_to_unique_values(lst, epsilon):
    for elem in lst:
        if lst.count(elem) > 1:
            indices = [i for i, x in enumerate(lst) if x == elem]
            toggle = 1
            factor = 1
            for index in indices:
                lst[index] = elem + epsilon*toggle*factor
                toggle = toggle *-1
                if toggle == 1:
                    factor = factor + 1
                    
def render_mat_to_unique_values(mat, epsilon, excluded_column):
    list_of_columns = []
    for column in range(mat.shape[1]):
        column_list = mat[:,column].tolist()
        if (column != excluded_column):
            render_list_to_unique_values(column_list, epsilon)
        list_of_columns.append(column_list)
    new_mat = np.array(list_of_columns)
    return np.transpose(new_mat)
    
def main(training_data_percentage):
    mat = np.genfromtxt("wdbc.data.txt", delimiter=',')
    training_mat_size =  int(mat.shape[0]*training_data_percentage)
    
    nof_shuffles = 20
    for depth_limit in range(1,21):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Depth limit: " + str(depth_limit))
        cumulative_test_success_per_shuffle = 0
        for i in range(nof_shuffles):
            #print("Shuffle number " + str(i))
            shuffled_mat = shuffle_array(mat)
            training_mat = shuffled_mat[0:training_mat_size]
            training_mat = render_mat_to_unique_values(training_mat, EPS,1)
            test_mat = shuffled_mat[training_mat_size:mat.shape[0]]
            
            root = Node(training_mat)
            
            available_feature_indices = dict(zip([i for i in range(mat.shape[1])], [True for i in range(mat.shape[1])]))
            available_feature_indices[1] = False
            time1 = datetime.now()
            #############################
            train_tree(root,depth_limit, available_feature_indices)
            #############################
            time2 = datetime.now()
            delta = time2-time1
            #print("Time difference for training: " + str(delta.total_seconds()))
        
            test_nof_rows = test_mat.shape[0]
            nof_successes = 0
            for row in range(test_nof_rows):
                actual_label = test_mat[row][1]
                predicted_label = get_label(root,test_mat[row])
                if actual_label == predicted_label:
                    nof_successes = nof_successes + 1
            percentage_of_success = 100 * (nof_successes / test_nof_rows)
            cumulative_test_success_per_shuffle = cumulative_test_success_per_shuffle + percentage_of_success
            #print("percentage of success in test: " + str(percentage_of_success))
            
            training_nof_rows = training_mat.shape[0]
            nof_successes = 0
            for row in range(training_nof_rows):
                actual_label = training_mat[row][1]
                predicted_label = get_label(root,training_mat[row])
                if actual_label == predicted_label:
                    nof_successes = nof_successes + 1
            percentage_of_success = 100 * (nof_successes / training_nof_rows)
            #print("percentage of success in training: " + str(percentage_of_success))
        print("Average test success: " + str(cumulative_test_success_per_shuffle / nof_shuffles))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return root
    