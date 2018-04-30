# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:33:11 2018

@author: tb945172
"""
import time

def maxush(a,b):
    if a>b:
        return a
    return b


def max_of_three(a,b,c):
    t=time.time()
    a= maxush(maxush(a,b),c)
    print(time.time()-t)
    return a

def length(a):
    l = 0
    for c in a:
        l = l+1
    return l

def is_vowel(a):
    if a in "aeiou":
        return True
    return False
    
        