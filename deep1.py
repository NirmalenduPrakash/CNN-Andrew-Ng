import numpy as np
from nnUtil import *
import matplotlib.pyplot as plt
import math

def L_modelBackward(AL,Y,z,w,A):
    L=len(w)
    grads={}
    dAL=-(np.divide(Y,AL))+(np.divide(1-Y,1-AL))
    dzL=sigmoidBackward(dAL,z[str(L)])
    grads["dw"+str(L)],grads["db"+str(L)],grads["da"+str(L-1)]=linearBackward(dzL,A[str(L-1)],w[str(L)])    
    for l in reversed(range(1,L)):
        grads["dz"+str(l)]=reluBackward(grads["da"+str(l)],z[str(l)])
        grads["dw"+str(l)],grads["db"+str(l)],grads["da"+str(l-1)]=linearBackward(grads["dz"+str(l)],A[str(l-1)],w[str(l)])
    return grads    

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

def two_layer_model(X,Y,layer_dims,learning_rate=.0075,num_iterations=3000):
    parameters=initialize_parameters_deep(layer_dims)
    costs=[]
    for i in range(0,num_iterations):
        w1,b1,w2,b2=parameters["w1"],parameters["b1"],parameters["w2"],parameters["b2"]
        z1=linearForward(X,w1,b1)
        A1=linearActivationForward(z1,"Relu")
        z2=linearForward(A1,w2,b2)
        A2=linearActivationForward(z2,"Sigmoid")
        cost=computeCost(A2,Y.T)
        costs.append(cost)
        z={}
        w={}
        A={}
        z["1"]=z1
        z["2"]=z2
        w["1"]=w1
        w["2"]=w2
        A["0"]=X
        A["1"]=A1
        A["2"]=A2    
        grads=L_modelBackward(A2,Y.T,z,w,A)
        parameters=updateParameters({"w1":w1,"w2":w2,"b1":b1,"b2":b2},grads,.0075)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    return parameters

def L_layer_model(X,Y,layer_dims,learning_rate=.0075,num_iterations=3000):
    parameters=initialize_parameters_deep(layer_dims)
    costs=[]
    w={}
    b={}
    z={}
    A={}
    A["0"]=X
    L=int(len(parameters)/2)
    for i in range(0,num_iterations):
        for l in range(1,L+1):
            w[str(l)]=parameters["w"+str(l)]
            b[str(l)]=parameters["b"+str(l)]
            z[str(l)]=linearForward(A[str(l-1)],w[str(l)],b[str(l)])
            if l<L:
                A[str(l)]=linearActivationForward(z[str(l)],"Relu")
            else:
                A[str(l)]=linearActivationForward(z[str(l)],"Sigmoid")
                cost= computeCost(A[str(l)],Y.T)
                costs.append(cost)           
        grads=L_modelBackward(A[str(L)],Y.T,z,w,A)
        parameters=updateParams_L(parameters,grads,.0075)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    return parameters

def calculateAccuracy(AL,Y):
    AL[AL<0.5]=0
    AL[AL>=0.5]=1
    return np.sum(AL==Y)/Y.shape[0]

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1)   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.
parameters=L_layer_model(train_x,train_y,(train_x.shape[1],20,7,5,1),num_iterations=3000)
A={}
w={}
b={}
z={}
A["0"]=test_x
L=4
for l in range(1,5):
    w[str(l)]=parameters["w"+str(l)]
    b[str(l)]=parameters["b"+str(l)]
    z[str(l)]=linearForward(A[str(l-1)],w[str(l)],b[str(l)])
    if(l<L):
        A[str(l)]=linearActivationForward(z[str(l)],"Relu")
    else:
        A[str(l)]=linearActivationForward(z[str(l)],"Sigmoid")
# w1,b1,w2,b2=parameters["w1"],parameters["b1"],parameters["w2"],parameters["b2"]
# z1=linearForward(test_x,w1,b1)
# A1=linearActivationForward(z1,"Relu")
# z2=linearForward(A1,w2,b2)
# A2=linearActivationForward(z2,"Sigmoid")
# print(calculateAccuracy(A2,test_y.T))
print(calculateAccuracy(A[str(4)],test_y.T))


