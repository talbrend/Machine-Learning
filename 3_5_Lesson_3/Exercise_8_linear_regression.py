# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]
y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]

ln_y = np.log(y)
y_arr = np.array(ln_y)


x_lists = []
for i in range(len(x)):
    x_lists.append([1,x[i]*x[i],x[i]])

x_matrix = np.array(x_lists)

x_t_x = np.matmul(np.transpose(x_matrix),x_matrix)
x_inv = np.linalg.inv(x_t_x)

final_x = np.matmul(x_inv,np.transpose(x_matrix))

h = np.matmul(final_x,y_arr)