# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:38:57 2024

@author: Zainon
"""
import  pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import LabelEncoder
from  sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
data=pd.read_csv("drug200.csv")
print(data)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_object=x.select_dtypes(include=["object"])
x_nonobject=x.select_dtypes(exclude=["object"])


la=LabelEncoder()

for i in range(x_object.shape[1]):
    x_object.iloc[:,i]=la.fit_transform(x_object.iloc[:,i])
X=pd.concat([x_nonobject,x_object],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

model=MLPClassifier(hidden_layer_sizes=(
    60,100,600),activation="logistic", learning_rate="constant",learning_rate_init=0.0001,max_iter=1000)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)

print("train_Acc is ", model.score(x_train,y_train))
print("test_Acc  is ", model.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))



rebort=classification_report(y_test,y_pred)

model_filename = 'NeuralNetwork_modelsqmound.joblib'
joblib.dump(model, model_filename)
print(f"تم حفظ النموذج في {model_filename}")


