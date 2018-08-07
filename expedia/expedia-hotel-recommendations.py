import numpy as np
import csv
from nnUtil import *

# import data
destinations_data=load_dataset_csv('./expedia/destinations.csv')
split_files={0:'train_0.csv'}
crossval_data=[]
readlimit=5000
counter=0
with open('./expedia/train.csv') as file:
    row=file.readline()
    train_data=np.zeros((4000,len(row)+destinations_data.values.shape[1]),dtype=None)
    for row in file:
        train_data[counter,0:len(row)]=row
        temp=destinations_data.values[destinations_data.values[:,0]==row[16]]
        if(len(temp)>0):
            train_data[counter,len(row):train_data.shape[1]]=temp
            
