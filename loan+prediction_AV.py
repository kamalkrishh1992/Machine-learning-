# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 08:07:11 2016

@author: Admin
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:\\dataset\\loandataset')
df = pd.read_csv('train.csv')

df.apply(lambda x: sum(x.isnull()),axis=0) 

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].fillna('No',inplace=True)

df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'].fillna(360,inplace=True)
df['Dependents'].fillna(0,inplace=True)





#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#var_mod = ['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
#for i in var_mod:    
 #   le.fit(df[i])
  #  df[i]=le.transform(df[i]) 


#le.fit(df["Dependents"])
#df["Dependents"]=le.transform(df["Dependents"]) 

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
df['LoanAmount'].hist(bins=20)


df.boxplot(column='ApplicantIncome', by = 'Gender')


df.loc[(df["Gender"]=="Male") & (df["Education"]=="Graduate") & (df["Loan_Status"]=="N"), ["Gender","Education","Loan_Status"]]

def num_missing(x):
  return sum(x.isnull())
  
df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
from scipy.stats import mode
mode(df['Gender'])
mode(df['Gender']).mode[0]



var_mod = ['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
for i in var_mod:
    df[i] = df[i].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
df['Credit_History'].fillna(1,inplace=True)

X=df.Credit_History[1:300].to_frame()
Y=df.Loan_Status[1:300].to_frame()
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
Z=df.Credit_History[300:].to_frame()

y=logreg.predict(Z)

        