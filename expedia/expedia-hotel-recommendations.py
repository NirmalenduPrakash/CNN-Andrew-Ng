import numpy as np
import csv
from nnUtil import *

# import data
destinations_data=load_dataset_csv('./expedia/destinations.csv')
split_files={0:'train_0.csv'}
crossval_data=[]
readlimit=5000
probab=np.random.rand(readlimit,1)<0.90
counter=0
with open('./expedia/train.csv','r') as file:
    reader=csv.reader(file,delimiter=',')
    for row in reader:
        if(counter==0):
            train_data=np.zeros((4500,len(row)+destinations_data.values.shape[1]-1),dtype=object) 
            counter+=1
            continue              
        temp=destinations_data.values[destinations_data.values[:,0]==row[16],1:]
        if(probab[counter-1]):
            train_data[counter-1,0:len(row)]=row
            if(len(temp)>0):
                train_data[counter-1,len(row):train_data.shape[1]]=temp
        else:
            crossval_data.append(np.r_[row,temp])
        counter+=1
        if(counter==readlimit):
            np.savetxt(split_files[len(split_files)-1],train_data,delimiter=',',fmt='%s')
            split_files.update({len(split_files):'train_'+str(len(split_files))+'.csv'})
            train_data=np.zeros((4500,len(row)+destinations_data.values.shape[1]-1),dtype=object)
            counter=0       
np.savetxt('crossval.csv',crossval_data,delimiter=',',fmt='%s')

            


