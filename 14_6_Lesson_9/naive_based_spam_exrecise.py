# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:17:03 2018

@author: tb945172
"""
import os
import math

def get_flat_list_from_list_of_lists(l):
    return [item for sublist in l for item in sublist]

def get_messages_from_directory_as_lists(list_of_files,directory):
    l = []
    for file_name in list_of_files:
        path = os.path.join(directory,file_name)
        l.append(get_words_from_file_as_list(path))
    return  l

def get_words_from_directory_as_list(list_of_files,directory):
    l = get_messages_from_directory_as_lists(list_of_files,directory)
    return  get_flat_list_from_list_of_lists(l)

def get_words_from_file_as_list(path):
    f = open(path)
    file_as_string = f.read()
    f.close()
    return file_as_string.split()

def map_word_to_frequency(list_of_words):
    dictionary = {}
    for word in list_of_words:
        frequency = dictionary.get(word)
        if frequency == None:
            dictionary[word] = 1
            continue
        else:
            dictionary[word] = frequency + 1
    return dictionary

def map_word_to_probabilty(dictionary,nof_words):
    ret_dict = {}
    for word in dictionary.keys():
        ret_dict[word] = dictionary[word] / nof_words
    return ret_dict

def get_list_of_probabilities(message,mapped_word_to_probability):
    l = []
    for word in message:
        prob = mapped_word_to_probability.get(word)
        if prob != None:
            l.append(prob)
    return l

def sum_list_elements_with_log(l):
    sum = 0
    for elem in l:
       sum = sum +  math.log(elem)
    return sum

def calc_prediction(training_spam_prob_dictionary, training_ham_prob_dictionary,message, training_spam_nof_words, training_ham_nof_words):
    list_of_probabilities_per_spam = get_list_of_probabilities(message,training_spam_prob_dictionary)
    list_of_probabilites_per_ham = get_list_of_probabilities(message,training_ham_prob_dictionary)
    
    prob_spam = training_spam_nof_words * sum_list_elements_with_log(list_of_probabilities_per_spam)
    prob_ham = training_ham_nof_words * sum_list_elements_with_log(list_of_probabilites_per_ham)
    
    if prob_spam > prob_ham:
        return 0
    return 1
    
training_spam_list_of_files = os.listdir("spam-train")
training_ham_list_of_files = os.listdir("nonspam-train")
test_spam_list_of_files = os.listdir("spam-test")
test_ham_list_of_files = os.listdir("nonspam-train")

training_spam_all_words =  get_words_from_directory_as_list(training_spam_list_of_files,"spam-train")
training_ham_all_words = get_words_from_directory_as_list(training_ham_list_of_files,"nonspam-train")
test_spam_all_words = get_words_from_directory_as_list(test_spam_list_of_files,"spam-test")
test_ham_all_words = get_words_from_directory_as_list(test_ham_list_of_files,"nonspam-train")

training_spam_dictionary = map_word_to_frequency(training_spam_all_words)
training_ham_dictionary = map_word_to_frequency(training_ham_all_words)
test_spam_dictionary = map_word_to_frequency(test_spam_all_words)
test_ham_dictionary = map_word_to_frequency(test_ham_all_words)

training_spam_nof_words = len(training_spam_all_words)
training_ham_nof_words = len(training_ham_all_words)
test_spam_nof_words = len(test_spam_all_words)
test_ham_nof_words = len(test_ham_all_words)

training_spam_prob_dictionary = map_word_to_probabilty(training_spam_dictionary, training_spam_nof_words)
training_ham_prob_dictionary = map_word_to_probabilty(training_ham_dictionary, training_ham_nof_words)
test_spam_prob_dictionary = map_word_to_probabilty(test_spam_dictionary, test_spam_nof_words)
test_ham_prob_dictionary = map_word_to_probabilty(test_ham_dictionary, test_ham_nof_words)

training_spam_messages_as_list = get_messages_from_directory_as_lists(training_spam_list_of_files,"spam-train")
training_ham_messages_as_list = get_messages_from_directory_as_lists(training_ham_list_of_files,"nonspam-train")
test_spam_messages_as_list = get_messages_from_directory_as_lists(test_spam_list_of_files,"spam-test")
test_ham_messages_as_list = get_messages_from_directory_as_lists(test_ham_list_of_files,"nonspam-train")

nof_correct_predictions = 0





