# 100 numpy exercises

# Please solve the following excercises
# Good Luck!

#### 1. Import the numpy package under the name `np`



#### 2. Print the numpy version
#### hint:  search on Google "numpy version", the first entry.


#### 3. Create a vector of zeros with the size 10
#### hint: use np.zeros()


#### 4.  How to find the memory size of any array
#### hint:  itemsize returns the size in memory of each element


#### 5.  How to get the documentation of the numpy add function from the IPython console?
#### hint:  try "?" before the function name


#### 6.  Create a vector of zeros with the size 10 but the fifth value which is 1 



#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)
#### hint: arange


#### 8.  Reverse a vector (first element becomes last) (★☆☆)
#### hint:  [::-1]


#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
#### hint: reshape


#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)
#### hint: nonzero


#### 11. Create a 3x3 identity matrix (★☆☆)
#### hint: the best numpy documentation is in the scipy website :)
#### try https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html


#### 12. Create a 3x3x3 array with random values (★☆☆)
#### hint:  https://docs.scipy.org/doc/numpy/reference/routines.random.html


#### 13. Create a 10x10 array with random values and find the minimum and maximum values 
#### hint:  a.min,a.max 

#### 14. Create a random vector of size 30 and find the mean value 
#### hint: a.mean 


#### 15. Create a 2d array with 1 on the border and 0 inside 
#### hint: ones()
#### hint2:  for an array a, which elements are chosen by a[1:-1] ?

#### 16. How to add a border (filled with 0's) around an existing array? 
#### hint: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html


#### 17. What is the result of the following expression? (★☆☆)


```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```
#### hint 1: nan is not a number
#### hint 2: try printing the expressions to see what is printed

#### 18. Create a 5x5 matrix with values 1,2,3,4,7 on the diagonal 
#### hint: np.diag


#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern 
#### hint1:  slicing an array you can use - list[start:end:step]
#### hint2:  [::2] - for even.  [1::2] - for odd


#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
#### hint:  https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.unravel_index.html


#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
#### hint:  numpy.tile(A, reps)
#### Construct an array by repeating A the number of times given by reps.


#### 22. Normalize a 5x5 random matrix
#### hint: create a random 5*5 matrix
#### subtract the min value, then devide by (max-min)


#### 23. Create an array of 2x4 with dtype numpy.int16, print the dtype of the array (RGBA)



#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) 
#### hint: numpy.dot()


#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. 


answer:  
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)


#### 26. What is the output of the following script? (★☆☆)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. Consider an integer vector Z, which of these expressions are legal?


```python
Z**Z
Z <- Z
1j*Z
```

#### 28. What are the result of the following expressions?


```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
```

#### 29. How to round away from zero a float array ? (★☆☆)
#### hint:  
#### 1.np.copysign
#### 2.np.ceil

#### 30. How to find common values between two arrays? (★☆☆)
#### hint:  https://docs.scipy.org/doc/numpy/reference/generated/numpy.intersect1d.html

