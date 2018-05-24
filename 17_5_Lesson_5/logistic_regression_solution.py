import numpy as np
import scipy.stats as stats 
from matplotlib import pyplot as plt

#np.random.seed(5)

def generate_random_clusters(n_points_in_cluster,n_clusters, std):
    center = np.random.rand(2)*4
    angles = 2*np.pi* np.linspace(0,1,n_clusters+1)[:-1]
    centers = np.c_[np.cos(angles),np.sin(angles)] + center
    noise = np.random.normal(0,std,(n_clusters,n_points_in_cluster,2))
    points = np.repeat(np.expand_dims(centers,1),n_points_in_cluster,axis=1)
    return points + noise

n_points_in_cluster = 99
n_clusters = 2
std = 0.8
data = generate_random_clusters(n_points_in_cluster,n_clusters, std)
data_points=data.reshape(-1,2)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, n_clusters))

nn=n_points_in_cluster

for i in range(n_clusters):
    plt.scatter(data_points[nn*i:nn*(i+1),0],data_points[nn*i:nn*(i+1),1],color=colors[i])

y=np.repeat(np.array(range(n_clusters)),n_points_in_cluster)
X = np.c_[data_points,np.ones(data_points.shape[0])]

def split_train_test(X,y,percentage_test):
    per_index=int(len(y)*(1-percentage_test))
    return X[:per_index,...],X[per_index:,...],y[:per_index],y[per_index:]

percentage_test = 0.2
indices= np.array(range(n_points_in_cluster*n_clusters))
np.random.shuffle(indices)

X_train, X_test, y_train, y_test = split_train_test(X[indices,:],y[indices],percentage_test)

def logit(x):
    return 1/(1+ np.exp(-x))

#def softmax(x):
#    exp_norm = np.exp(x-x.max())
#    return exp_norm/exp_norm.sum(axis=1)
              
EPS = 1e-7

def minus_log_likelihood(res,y):
    return -(y*np.log(res+EPS)+(1-y)*np.log(1-res+EPS)).mean()

def predict_logit(x, teta):
    return logit(np.dot(x,teta))>0.5

#def predict_softmax(x, teta):
#    return softmax(np.dot(x,teta)).argmax()


def run_gradient_descent_logit(X,y,start,rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        res=logit(np.dot(X,t))
        loss=minus_log_likelihood(res,y)
        grad=np.dot((res-y),X)/len(X)
        t=t-rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t
        
start =  np.append(np.random.normal(0,0.1,(2)),0)
print("start: " + str(start))
teta = run_gradient_descent_logit(X_train,y_train,start,0.1,100)
train_precision=(predict_logit(X_train, teta)==y_train).sum()/len(y_train)
test_precision=(predict_logit(X_test, teta)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)

    
def create_circles(n_points_in_cluster,n_clusters, std):
    angles = 2*np.pi*np.random.rand(n_clusters,n_points_in_cluster)
    radii =np.array([stats.truncnorm.rvs(-0.1+i,0.1+i,size=(n_points_in_cluster)) for i in range(2,2+n_clusters)])
    return np.swapaxes(np.swapaxes(np.c_[[radii*np.cos(angles),radii*np.sin(angles)]],0,2),0,1)


    