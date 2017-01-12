# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 07:08:31 2016

@author: Admin
"""
from __future__ import print_function


import datetime as dt

import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

import os
os.chdir('D:\\dataset')
retail = pd.read_excel('OnlineRetail.xlsx') 

retail['Purchaseamount']=retail['Quantity']*retail['UnitPrice']

m=retail[["CustomerID" ,"Purchaseamount"]].groupby(['CustomerID'],as_index=False).sum()


f=retail[["CustomerID" ,"InvoiceDate"]].groupby(['CustomerID'])['InvoiceDate'].nunique()
F=f.tolist()

m["feq"]=F

r=retail[["CustomerID" ,"InvoiceDate"]].groupby(['CustomerID'],as_index=False).max()



r["today"]=[dt.datetime(2011,12,9)]*len(r)

r["recency"]=(r["today"]-r["InvoiceDate"])

r["recency"]=((r["recency"]) / np.timedelta64(1, 'D')).astype(int)

m["recency"]=r["recency"]

rfm =m

d= m.quantile(q=[0.20,0.4,0.50,0.60,0.80])

AmountClass=[]
for i in range(0,len(m)):
 if m.Purchaseamount[i]<=234.0:
    AmountClass.append(1)
 elif m.Purchaseamount[i]<=465.0:
    AmountClass.append(2)
 elif m.Purchaseamount[i]<=648.0:
    AmountClass.append(3)
 elif m.Purchaseamount[i]<=909.0: 
    AmountClass.append(4)
 else:
    AmountClass.append(5)

m['ClassofAmount']=AmountClass   
    
D=[]
for i in range(0,len(m)):
 if m.feq[i]<=1:
    D.append(1)
 elif m.feq[i]<=2:
    D.append(2)
 elif m.feq[i]<=3:
    D.append(3)
 elif m.feq[i]<=4: 
    D.append(4)
 else:
    D.append(5)
m['ClassofFreq']=D


A=[]
for i in range(0,len(m)):
 if m.recency[i]<=10:
    A.append(5)
 elif m.recency[i]<=30:
    A.append(4)
 elif m.recency[i]<=49:
    A.append(3)
 elif m.recency[i]<=70: 
    A.append(2)
 else:
    A.append(1)
m['Classofrecency']=A
    
m['f'] = np.log(m['feq']+1)
m['r'] = np.log(m['recency']+1)
m['m'] = np.log(m['Purchaseamount']+1)



# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)

m['RFMClass'] = m.Classofrecency.map(str) \
                            + m.ClassofFreq.map(str) \
                            + m.ClassofAmount.map(str)

m['RFMClass'].astype(str)

from sklearn.cluster import KMeans
import numpy as np
X = m[['r','f','m']].as_matrix()
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
kmeans.labels_
m['y']=kmeans.predict(X) 
kmeans.cluster_centers_

silhouette_avg = silhouette_score(X, m['y'])

while i<len(m):
    if m['m'][i]<0:
        m['m'][i]=0
    elif m['m'][i]>1000000:
        m['m'][i]=10000
 
while i<len(m):
    if m['r'][i]<0:
        m['r'][i]=0
    elif m['r'][i]>10000:
        m['r'][i]=1000   

while i<len(m):
    if m['f'][i]<0:
        m['f'][i]=0
    elif m['f'][i]>100:
        m['f'][i]=100
        

df1 = m[['recency','feq','Purchaseamount','y']]
        
low1_cust=df1.loc[df1['y'] == 6]
low2=df1.loc[df1['y'] == 5]
premium_cust=df1.loc[df1['y'] == 4]        
low_cust=df1.loc[df1['y'] == 3]
loyal=df1.loc[df1['y'] == 2]
leads=df1.loc[df1['y'] == 1]
average=df1.loc[df1['y'] == 0]




