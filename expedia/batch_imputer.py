import numpy as np
import math
from decimal import Decimal

def calculateCost(h,y):
    m=y.shape[0]
    return (1/2*m)*np.sum(np.square(h-y))

def forwardProp(x,w,b):
    h=x.dot(w)+np.repeat(b,x.shape[0]).reshape(-1,1)
    for i in range(0,h.shape[0]):
        h[i,0]:Decimal(h[i,0])
    return h

def backProp(h,y,x):
    m=x.shape[0]
    return (1/m)*np.sum((h-y)*x,axis=0).reshape(-1,1),(1/m)*np.sum(h-y)

def initialize_params(feature_cnt):
    w=np.random.randn(feature_cnt).reshape(-1,1)
    b=0
    return w,b

def process(x,w,b,y):
    h=forwardProp(x,w,b)
    cost=calculateCost(h,y)
    dw,db=backProp(h,y,x)
    return dw,db,cost

def updateParameters(w,b,dw,db,learning_rate):
    w-=learning_rate*dw
    b-=learning_rate*db
    return w,b    
