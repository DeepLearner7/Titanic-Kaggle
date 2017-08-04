# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:46:12 2017

@author: Saurabh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:42:36 2017

@author: Saurabh
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:27:50 2017

@author: Saurabh
"""


import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.Sex = np.where(train.Sex=='male',1,0)

def clean(z):
    return z[0]

def con(p):
    if(p>60):
        return 2
    elif(p<10):
        return 0
    else:
         return 1
    

train.Cabin.fillna(method='ffill',inplace=True)
train.Cabin.fillna(method='bfill',inplace=True)

j=0
for i in train.Cabin:
    train.Cabin[j] = clean(i)
    j=j+1
    
train.Embarked.fillna('Q',inplace=True)
train['Cabin'] = train.Cabin.astype("category").cat.codes
train['Embarked'] = train.Embarked.astype("category").cat.codes
test['Embarked'] = test.Embarked.astype("category").cat.codes



feature_col = ['Pclass','Sex','Fare','Age','SibSp','Cabin','Parch','Embarked']
training = pd.DataFrame(train[feature_col])
testing = pd.DataFrame(test[feature_col])

training.Cabin.fillna(method='ffill',inplace=True)
training.Cabin.fillna(method='bfill',inplace=True)


testing.Cabin.fillna(method='ffill',inplace=True)
testing.Cabin.fillna(method='bfill',inplace=True)

j=0
for i in testing.Cabin:
    testing.Cabin[j] = clean(i)
    j=j+1

j=0
for i in training.Age:
    training.Age[j] = con(i)
    j=j+1
    
j=0
for i in testing.Age:
    testing.Age[j] = con(i)
    j=j+1

testing['Cabin'] = testing.Cabin.astype("category").cat.codes
training['Cabin'] = training.Cabin.astype("category").cat.codes
    
testing.Fare = testing.Fare.fillna(method='ffill')
testing.Age = testing.Age.fillna(np.mean(testing.Age))
testing.Sex = np.where(testing.Sex=='male',1,0)

training.Fare = training.Fare.fillna(method='ffill')
training.Age = training.Age.fillna(np.mean(training.Age))

rf = RandomForestClassifier(n_estimators=1000,oob_score=True)
rf.fit(training,train.Survived)
pred = rf.predict(testing)

final = pd.DataFrame()

final['Survived'] = pred 
final['PassengerId'] = test.PassengerId

final.to_csv("output_titanic_randomForest.csv",sep=',',index=False)


