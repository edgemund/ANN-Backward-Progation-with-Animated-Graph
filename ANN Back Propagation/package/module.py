"""*******************************
||| @author: NGUYEN DINH HAI    |||
||| @Version: 1.3               |||
||| @sice: Nov 11, 2019         |||
|||*****************************|||
"""

import numpy as np
import matplotlib.pyplot as plt

#softmax function
def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),\
        (y, np.arange(len(y)))), \
        shape = (C, len(y))).toarray()
    return Y

# cost or loss function
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

# transform data for iris
def transform_irisdata(pathx, pathy, part, numberpath = 5):
    c = 1 #count the mumber class
    # Load data
    X_data_f = np.loadtxt(pathx)
    Y = np.loadtxt(pathy)
    X_data = np.zeros_like(X_data_f)
    max = np.max(X_data_f, axis = 0)
    min = np.min(X_data_f, axis = 0)
    for i in range(X_data_f.shape[0]) :
        for j in range(X_data_f.shape[1]):
            X_data[i,j] = \
            (X_data_f[i,j]-min[j])/(max[j]-min[j])
    #print(X_data)
    y_data = np.zeros(Y.shape[0], dtype='uint8')
    for i in range(y_data.shape[0]):
        y_data[i] = int(Y[i])
        if(i>0)and(y_data[i]!=y_data[i-1]) :
            c += 1
    # caculater the element in the part
    element = int(y_data.shape[0]/numberpath)
    # Data for testing
    y_t = y_data[(part-1)*element:part*element]
    X_t = X_data_f[(part-1)*element:part*element,:].T
    #Data for training
    X = np.concatenate\
    ((X_data[:(part-1)*element,:],X_data[part*element:,:]), axis=0).T
    #print(X)
    y = np.concatenate\
    ((y_data[:(part-1)*element],y_data[part*element:]),axis = None)
    X_data = X_data_f
    return X_data, y_data, c, X, y, y_t, X_t
def transform(pathx, pathy, part, numberpath = 5):
    c = 1 #count the mumber class
    # Load data
    X_data_f = np.loadtxt(pathx)
    Y = np.loadtxt(pathy)
    X_data = np.zeros_like(X_data_f)
    max = np.max(X_data_f, axis = 0)
    min = np.min(X_data_f, axis = 0)
    avg = np.mean(X_data_f, axis = 0)
    for i in range(X_data_f.shape[0]) :
        for j in range(X_data_f.shape[1]):
            X_data[i,j] = \
            (X_data_f[i,j]-avg[j])/(max[j]-min[j])
    #print(X_data)
    y_data = np.zeros(Y.shape[0], dtype='uint8')
    for i in range(y_data.shape[0]):
        y_data[i] = int(Y[i])
        if(i>0)and(y_data[i]!=y_data[i-1]) :
            c += 1
    # caculater the element in the part
    element = int(y_data.shape[0]/numberpath)
    # Data for testing
    y_t = y_data[(part-1)*element:part*element]
    X_t = X_data[(part-1)*element:part*element,:].T
    #Data for training
    X = np.concatenate\
    ((X_data[:(part-1)*element,:],X_data[part*element:,:]), axis=0).T
    #print(X)
    y = np.concatenate\
    ((y_data[:(part-1)*element],y_data[part*element:]),axis = None)
    X_data = X_data_f
    return X_data, y_data, c, X, y, y_t, X_t
def display(X,Y, axs,title,xlabel,ylabel) :
    for i in range (Y.shape[0]):
        axs.set_title(title)
        axs.set_xlabel(xlabel); axs.set_ylabel(ylabel)
        if Y[i] == 0 :
            axs.plot(X[i,0],X[i,1],'go', label ="Class 0")
        if Y[i] == 1 :
            axs.plot(X[i,0],X[i,1],'r*', label = "Class 1")
        if Y[i] == 2 :
            axs.plot(X[i,0],X[i,1],'b^', label = "Class 2")
