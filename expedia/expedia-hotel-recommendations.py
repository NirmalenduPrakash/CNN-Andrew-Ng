import numpy as np
import csv
from nnUtil import *
from datetime import date 
import batch_imputer_tf as imputer
from batch_imputer import *
from matplotlib import pyplot as plt

# import data
# destinations_data=load_dataset_csv('./expedia/destinations.csv')

# PCA to reduce destination laterals to 1 dim
# x=np.array(destinations_data.values[1:,1:],dtype=float)
# x=normalize(x)
# cov=calcCovariance(x)
# eigen=calcEigen(cov)[1]
# x_reduced=getReducedMatrix(x,1,eigen)
# destinations_data_reduced=np.c_[np.array(destinations_data.values[1:,0],dtype=float),x_reduced]

# split_files={0:'train_0.csv'}
# crossval_data=[]
# train_data=[]
# readlimit=50000
# probab=np.random.rand(readlimit,1)<0.95
# counter=0
# file=open('./expedia/train.csv','r')
# reader=csv.reader(file,delimiter=',')
# for row in reader:
#     if(counter==0):
#         counter+=1
#             continue    
#         if(row[0]!=''):
#             row[0]=row[0].split('-')[1]
#         if(row[11]!=''):
#             checkin_date=date(int(row[11].split('-')[0]),int(row[11].split('-')[1]),int(row[11].split('-')[2].split(' ')[0]))  
#             row[11]=row[11].split('-')[1]
#         if(row[12]!=''):
#             chkout_date=date(int(row[12].split('-')[0]),int(row[12].split('-')[1]),int(row[12].split('-')[2].split(' ')[0]))            
#             row[12]=(chkout_date-checkin_date).days
#         temp=destinations_data_reduced[destinations_data_reduced[:,0]==float(row[16]),1]
#         if(temp!=None):
#             row[16]=temp[0]
#         else:
#             row[16]=''    
#         if(probab[counter-1]):
#             train_data.append(row)    
#         else:   
#             crossval_data.append(row) 
#         if(counter==readlimit):
#             f= open('./expedia/'+split_files[len(split_files)-1],'w')
#             writer=csv.writer(f)
#             writer.writerows(train_data)
#             f.close()
#             split_files.update({len(split_files):'train_'+str(len(split_files))+'.csv'})
#             train_data=[]
#             counter=0
#         counter+=1               
#     file.close()
# f= open('./expedia/crossval.csv','w')
# writer=csv.writer(f)
# writer.writerows(crossval_data)
# f.close()

# impute orig_destination distance using linear regression
file_paths=[]
for i in range(0,753):
    file_paths.append('./expedia/train_'+str(i)+'.csv')
file_paths.append('./expedia/crossval.csv')    
feature_cnt=9
imp=imputer.batchMiceImputer(feature_cnt,file_paths)
# w,b,costs=imp.process()
# plt.plot(range(len(costs)),costs)
# plt.show()

# update missing orig-destination distance in train and crossval files
# imp.updateOrigDest()



