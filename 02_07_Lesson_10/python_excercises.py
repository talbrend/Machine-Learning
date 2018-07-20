# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:26:39 2018

@author: tb945172
"""

import sympy
from datetime import datetime
from datetime import date
import time
import timeit

# List comprehension
dict_of_squares = {i:i*i for i in range(1,100)}
list_of_squares = [i*i for i in range(1,100)]
list_of_primes = [i for i in range(100) if sympy.isprime(i)]

#Files

fo = open("hello.txt", "w+")
fo.write( 'Hello World');

# Close opend file
fo.close()

fr = open("hello.txt")
str_ = fr.read()
fr.close()
print("file: " + str_)

fw = open("squares.txt", "w+")
for item in list_of_squares:
    fw.write(str(item) + "\n")
fw.close()

#With

str_ = "Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition.[1] The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification.[2] The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set."

with open('output.txt', 'w+') as f:
     f.write(str_)
     f.seek(0)
     lines = f.readlines()
     print ("lines : " + str(lines))
     for line in lines:
        print(line)
        
#Strings
        
l = str_.split(" ")
new_str = " ".join(l)
print("new_str: \n" + new_str)
    
#Time

date_of_now = datetime.now()
day_before_yesterday =date(date_of_now.year,date_of_now.month,date_of_now.day-2)
print(day_before_yesterday)
if (date_of_now.hour+12 > 23):
    day = date_of_now.day + 1
else:
    day = date_of_now.day
datetime_in_twelve_hours = datetime(date_of_now.year,date_of_now.month,day,(date_of_now.hour+12)%24,date_of_now.minute, date_of_now.second)
print(datetime_in_twelve_hours)

time1 = datetime.now()
list_of_squares = [i*i for i in range(1,100)]
time2 = datetime.now()
delta = time2-time1
print(delta)
print("Time difference for comprehended list of squares: " + str(delta.total_seconds()))

time1 = datetime.now()
l=[]
for i in range(1,100):
    l.append(i*i)
time2 = datetime.now()
delta = time2-time1
print(delta)
print("Time difference for loop of squares: " + str(delta.total_seconds()))

timeit_ = timeit.timeit("[i*i for i in range(1,100)]",number=1)
print("Time difference for comprehended list of squares:" + str(timeit_))

