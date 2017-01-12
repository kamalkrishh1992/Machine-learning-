# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:58:00 2017

@author: Admin
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('D:\\dataset')
df = pd.read_csv('diamonds.csv')
df.head()
df = df.drop('Unnamed: 0', 1)

#pivot table to check median price by color and cut
table_median = df.pivot_table(values='price', index='cut',columns="color", aggfunc=np.median )
table_mean = df.pivot_table(values='price', index='cut',columns="color", aggfunc=np.mean )

#check null values
df.apply(lambda x: sum(x.isnull()),axis=0)

#check unique values
df.apply(lambda x: len(x.unique()))

#Filter categorical variables
categorical_columns = [x for x in df.dtypes.index if df.dtypes[x]=='object']
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (df[col].value_counts())
        
    
#One Hot Coding:
df = pd.get_dummies(df, columns=['cut'])    
df = pd.get_dummies(df, columns=['color'])  
df = pd.get_dummies(df, columns=['clarity']) 
df_train=df.loc[0:30000]
df_test=df.loc[30001:]

df_train1 = df_train.drop('price', 1)
df_train2 = df_train1.drop("depth",1)
df_train3 = df_train2.drop("table",1)
df_test1=df_test.drop('price',1)
df_test2=df_test1.drop("depth",1)
df_test3=df_test2.drop("table",1)

#Linear Regression
from sklearn import linear_model
regr = linear_model.LinearRegression()

#Regression using single variable
#Here carat explains 79% of the change in price
regr.fit(df_train.carat.to_frame(), df_train.price)##0.7917
RSquare_carat=regr.score(df_train1.carat.to_frame(), df_train.price)

#Here depth variable expainsonly one percent of the variaton in price
regr.fit(df_train.depth.to_frame(), df_train.price)#0.001
RSquare_depth=regr.score(df_train1.depth.to_frame(), df_train.price)

# Train the model using the training sets
regr.fit(df_train1, df_train.price)
RSquare_all_variables=regr.score(df_train1, df_train.price)###0.90753

#Predictions
df_test['predicted']= regr.predict(df_test1)
y=regr.predict(df_test1)

#Plotting actual vs predicted
plt.scatter(df_test.price, df_test.predicted)
 
regr.fit(df_train.carat.to_frame(), df_train.price)
RSquare3=regr.score(df_train1.carat.to_frame(), df_train.price)