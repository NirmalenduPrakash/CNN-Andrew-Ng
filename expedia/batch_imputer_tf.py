import tensorflow as tf
import numpy as np
from nnUtil import *
import csv
import math
from sklearn import preprocessing

class batchMiceImputer:
    def __init__(self,features_cnt,file_paths):
        self.x=tf.placeholder(tf.float64)
        self.y=tf.placeholder(tf.float64)
        self.m=tf.placeholder(tf.float64)
        self.w=tf.random_uniform(shape=(9,1),dtype=tf.float64,name="w")
        # self.w=tf.Variable(np.random.randn(features_cnt).reshape(-1,1),name="w")
        self.b=tf.Variable(0,dtype=tf.float64,name="b")
        self.files=file_paths

    def removeCorrelatedFeatures(self):
        for f in self.files:
            df=load_dataset_csv(f)
            df=pd.DataFrame(df.values[~np.any(np.isnan(df.values),axis=1)])
            # corr_matrix=df.corr().abs()
            # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
           
            # std_scaler=preprocessing.StandardScaler()
            # x_scaled=std_scaler.fit_transform(df)
            # df=pd.DataFrame(x_scaled)
            to_drop=df.var()

    # def update(self,files):
    #     y_model=tf.tensordot(normalize(self.x),self.w,1)+self.b
    #     for f in self.files:
    #         data=load_dataset_csv(f)
    #         x_val= data.values.astype(np.float)
    #         self.x=x_val[:,[1,2,3,4,5,20,21,22,23]]
    #         y_val=x_val[:,6]
    #         std=np.std(y_val)
    #         mean=np.mean(y_val)
    #         indices=np.isnan(y_val)           
    #         y_predict=tf.Session().run(y_model)
    #         np.put(x_val[:,6],indices,np.take((y_predict*std)+mean,indices))
    #         f= open(f,'w')
    #         writer=csv.writer(f)
    #         writer.writerows(x_val)
    #         f.close()

    def updateOrigDest(self):
        for f in self.files:
            data=load_dataset_csv(f).astype(float)
            avg=[]
            uniq_userReg_HotelReg=np.unique(data.values[:,[3,4,22]],axis=1)
            uniq_userCtry_HotelReg=np.unique(data.values[:,[3,22]],axis=1)            
                    
            for i in range(data.values.shape[0]):
                userReg_truth=((uniq_userReg_HotelReg[:,0]==data.values[i,3]) & (uniq_userReg_HotelReg[:,1]==data.values[i,4])
                    & (uniq_userReg_HotelReg[:,2]==data.values[i,22]))
                userCity_truth=(uniq_userCtry_HotelReg[:,0]==data.values[i,3]) & (uniq_userCtry_HotelReg[:,1]==data.values[i,22])
                if(math.isnan(data.values[i,6])):
                    dist=data.values[userReg_truth,6]
                    avg=np.mean(dist[~np.isnan(dist)])
                    if(math.isnan(avg)):
                        dist=data.values[userCity_truth,6]
                        avg=np.mean(dist[~np.isnan(dist)])
                    data.values[i,6]=avg

            np.savetxt(f,data.values,fmt='%d',newline=" ")
            # for i in range(uniq_userReg_HotelReg.shape[0]):
            #     dist=data.values[(data.values[:,3]==uniq_userReg_HotelReg[i,0]) & (data.values[:,4]==uniq_userReg_HotelReg[i,1])
            #         & (data.values[:,22]==uniq_userReg_HotelReg[i,2]),6]
            #     avg.append(np.mean(dist[~np.isnan(dist)]))
            # uniq_userReg_HotelReg=np.c_[uniq_userReg_HotelReg,avg]        



    def process(self):        
        y_model=tf.tensordot(self.x,self.w,1)+self.b
        error=tf.div(tf.reduce_sum(tf.square(self.y-y_model)),(tf.multiply(self.m,2)))
        train_op=tf.train.GradientDescentOptimizer(0.000001).minimize(error)
        model=tf.global_variables_initializer()
        costs=[]
        with tf.Session() as session:
            session.run(model)
            for f in self.files:
                data=load_dataset_csv(f)
                x_val= data.values.astype(np.float)
                x_val=x_val[~np.any(np.isnan(x_val),axis=1)]
                y_val=x_val[:,6]
                x_val=x_val[:,[1,2,3,4,5,20,21,22,23]]                
                x_val=normalize(x_val)                
                cost=session.run(error,feed_dict={self.x:x_val,self.y:y_val,self.m:y_val.shape[0]})
                session.run(train_op,feed_dict={self.x:x_val,self.y:y_val,self.m:y_val.shape[0]})
                costs.append(cost)
            w_value=session.run(self.w)
            b_value=session.run(self.b)    
        return w_value,b_value,costs    



