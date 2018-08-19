import numpy as np
import h5py
import pandas as pd
import math

def linearForward(A,w,b):
    z=np.dot(A,w.T)+b.T
    return z

def linearActivationForward(z,activation):
    if(activation=="Relu"):
        return np.maximum(0,z)
    elif(activation=="Sigmoid"):
        return 1/(1+np.exp(-z))

def forwardPropWithL2(X,y,parameters,layer_dims,lambd):
    A={'0':X}
    z={}
    for l in range(0,len(layer_dims)-1):
        z.update({str(l+1):linearForward(A[str(l)],parameters['w'+str(l+1)],parameters['b'+str(l+1)])})
        if(l==len(layer_dims)-2):
            A.update({str(l+1):linearActivationForward(z[str(l+1)],"Sigmoid")})
            cost=computeCost(A[str(l+1)],y,parameters,lambd,layer_dims)
        else:    
            A.update({str(l+1):linearActivationForward(z[str(l+1)],"Relu")})            
    return A,z,cost

def backwardPropWithL2(A,z,y,parameters,lambd,layer_dims):
    dz={}
    dw={}
    db={}
    da={}
    daL=(-y/A[str(len(A)-1)])+(1-y)/(1-A[str(len(A)-1)])   
    for l in reversed(range(1,len(layer_dims))):
        if(l==len(layer_dims)-1):
            dz[str(l)]=sigmoidBackward(daL,z[str(l)])
        else:
            dz[str(l)]=reluBackward(A[str(l)],z[str(l)])    
        dw[str(l)],db[str(l)],da[str(l-1)]=linearBackward(dz[str(l)],A[str(l-1)],parameters['w'+str(l)],lambd)
    return dw,db    

def computeCost(AL,Y,parameters,lambd,layer_dims):
    m=Y.shape[0]
    cost=(-1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))
    for l in range(1,len(layer_dims)):
        cost+=(lambd/(2*m))*np.sum(np.square(parameters['w'+str(l)]))
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

def linearBackward(dz,A_prev,w,lambd):
    m=A_prev.shape[0]
    dw=(1/m)*np.dot(A_prev.T,dz)+(lambd/m)*w.T
    db=(1/m)*np.sum(dz,axis=0,keepdims=True)
    da_prev=dz.dot(w)
    return dw,db,da_prev

def updateParameters(parameters,grads,learning_rate):
    L=int(len(parameters)/2)
    for l in range(L):
        parameters["w"+str(l+1)]=parameters["w"+str(l+1)]-learning_rate*grads["dw"][str(l+1)].T
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"][str(l+1)].T 
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
    return pd.read_csv(file,dtype=float)

def initialize_parameters_deep(layer_dims):    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * (1/math.sqrt(layer_dims[l]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###
        
        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))        
    return parameters      

def normalize(x):
    mean=np.mean(x,axis=0)
    std=np.std(x,axis=0)
    return (x-mean)/std

def calcCovariance(x):
    return np.cov(x.T)

def calcEigen(cov):
    return np.linalg.eig(cov)

def getReducedMatrix(x,k,eigen):
    return x.dot(eigen[:,0:k])    

    