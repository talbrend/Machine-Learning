# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:49:12 2017

@author: Eli
"""

import numpy as np

X=np.array([[31,22],[22,21],[40,37],[26,25]])
y=np.array([2,3,8,12])


def linear_regression(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)


def check_regression(X,y):
    res=linear_regression(X,y)
    print('Regression result: ',res)
    print('predicted y:', np.dot(X,res))
    print('loss :', ((np.dot(X,res)-y)**2).mean())

# 3D plane crossing (0,0,0)
check_regression(X,y)

# 3rd feature: constant
X1=np.ones((4,3))
X1[:,:-1]=X
check_regression(X1,y)

# 3rd feature: x1-x2
X2=X1.copy()
X2[:,2]=X[:,0]-X[:,1]
check_regression(X2,y)

# 3rd feature: (x1-x2)^2
X3=X1.copy()
X3[:,2]=(X[:,0]-X[:,1])**2
check_regression(X3,y)

# 3rd feature: (x1-x2)^2, 4th feature constant
X4=np.ones((4,4))
X4[:,:-1]=X3
check_regression(X4,y)


###########################
#### Gradient Descent #####
###########################

x=np.array([0,1,2,3],dtype=np.float32)
y=np.array([1,3,7,13],dtype=np.float32)
X=np.c_[np.ones_like(x),x,x**2]

# regression works nicely, but this is not part of the task
check_regression(X,y)


start = np.array([2,2,0],dtype=np.float32)

def my_model(t):
    return np.dot(X,t)

def mse_loss(res,y):
    return 0.5*((res-y)**2).mean()

def mse_loss_grad(res,y):
    return (np.dot(res-y,X))/len(X)
   

def run_gradient_descent(X,y,start,rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        res=np.dot(X,t)
        loss=0.5*((res-y)**2).mean()
        grad=(np.dot(res-y,X))/len(X)
        t=t -rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
        

#run_gradient_descent(X,y,start,0.01,100)


def run_momentum_gradient_descent(X,y,start,model_function,rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        calc=model_function(t)
        loss=mse_loss(calc,y)
        grad=mse_loss_grad(calc,y)
        v=momentum_decay*v - rate*grad
        t= t+v
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))


#run_momentum_gradient_descent(X,y,start,my_model,0.01,0.9,100)


def run_nesterov_momentum_gradient_descent(X,y,start,model,rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        calc=model(t)
        loss=mse_loss(calc,y)
        grad=mse_loss_grad(model(t+momentum_decay*v),y)
        v=momentum_decay*v - rate*grad
        t= t+v
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
        
        
run_nesterov_momentum_gradient_descent(X,y,start,my_model,0.01,0.9,100)



