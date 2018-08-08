import numpy as np
import h5py
import pandas as pd

def linearForward(A,w,b):
    z=np.dot(A,w.T)+b.T
    return z

def linearActivationForward(z,activation):
    if(activation=="Relu"):
        return np.maximum(0,z)
    elif(activation=="Sigmoid"):
        return 1/(1+np.exp(-z))

def computeCost(AL,Y):
    m=Y.shape[0]
    cost=(-1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))
    cost=np.squeeze(cost)
    assert(cost.shape==())
    return cost

def reluBackward(da,z):
    dz=np.array(da,copy=True)
    dz[z<=0]=0
    return dz

def sigmoidBackward(da,z):
    s=1/(1+np.exp(-z))
    dz=da*s*(1-s)
    return dz

def linearBackward(dz,A_prev,w):
    m=A_prev.shape[0]
    dw=(1/m)*np.dot(A_prev.T,dz)
    db=(1/m)*np.sum(dz,axis=0,keepdims=True)
    da_prev=dz.dot(w)
    return dw,db,da_prev

def updateParameters(parameters,grads,learning_rate):
    L=int(len(parameters)/2)
    for l in range(L):
        parameters["w"+str(l+1)]=parameters["w"+str(l+1)]-learning_rate*grads["dw"+str(l+1)].T
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)].T 
    return parameters       

def updateParams_L(parameters,grads,learning_rate):
    # w=parameters["w"]
    # b=parameters["b"]
    L=int(len(parameters)/2)
    for l in range(L):
        parameters["w"+str(l+1)]=parameters["w"+str(l+1)]-learning_rate*grads["dw"+str(l+1)].T
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)].T 
    return parameters     

def load_dataset():
    train_dataset = h5py.File('./train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_dataset_csv(file):
    return pd.read_csv(file,dtype=object)