import numpy as np
import csv
from nnUtil import *

# import data
destinations_data=load_dataset_csv('./expedia/destinations.csv')
split_files={0:'train_0.csv'}
crossval_data=[]
train_data=[]
readlimit=5000
probab=np.random.rand(readlimit,1)<0.90
counter=0
with open('./expedia/train.csv','r') as file:
    reader=csv.reader(file,delimiter=',')
    for row in reader:
        if(counter==0):
            # train_data=np.zeros((4500,len(row)+destinations_data.values.shape[1]-1),dtype=object) 
            counter+=1
            continue              
        temp=destinations_data.values[destinations_data.values[:,0]==row[16],1:]
        if(probab[counter-1]):
            # train_data[counter-1,0:len(row)]=row
            if(len(temp)>0):
                # train_data[counter-1,len(row):train_data.shape[1]]=temp
                train_data.append(np.array(np.r_[row,temp[0]],dtype=object))
            else:
                train_data.append(row)    
        else:
            if(len(temp)>0):
                crossval_data.append(np.array(np.r_[row,temp[0]],dtype=object))
            else:    
                crossval_data.append(row) 
        if(counter==readlimit):
            # np.savetxt(split_files[len(split_files)-1],train_data,delimiter=',',fmt='%s')
            f= open('./expedia/'+split_files[len(split_files)-1],'w')
            writer=csv.writer(f)
            writer.writerows(train_data)
            split_files.update({len(split_files):'train_'+str(len(split_files))+'.csv'})
            # train_data=np.zeros((4500,len(row)+destinations_data.values.shape[1]-1),dtype=object)
            train_data=[]
            counter=0
        counter+=1               
# np.savetxt('crossval.csv',crossval_data,delimiter=',',fmt='%s')
f= open('./expedia/crossval.csv','w')
writer=csv.writer(f)
writer.writerows(crossval_data)

#initialize params
layer_dims=(172,86,43,21,10,1)
parameters=initialize_parameters_deep(layer_dims) 
num_iterations=500
for i in range(1,num_iterations):
    for file in split_files:
        train_data=np.genfromtxt('./expedia/'+split_files[file],delimiter=',')
        forwardPropWithL2()


