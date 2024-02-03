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
from sklearn.metrics import confusion_matrix
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


list=[0.0001,0.001,0.01,0.1,0.5]
list2=[]
for i in range(len(list)):
    model=MLPClassifier(hidden_layer_sizes=( 60,100,600), learning_rate="constant",learning_rate_init=list[i],max_iter=1000)

    model.fit(x_train, y_train)

    scor=model.score(x_test,y_test)
    list2.append(scor)

  
print(list2)



model_filename = 'NeuralNetwork_modeltanh.joblib'
joblib.dump(model, model_filename)
print(f"تم حفظ النموذج في {model_filename}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",xticklabels=['DrugY', 'drugA','  drugB','  drugC ','drugX'], yticklabels=['DrugY', 'drugA','  drugB','  drugC ','drugX'])
plt.title('مصفوفة الالتباس (Confusion Matrix)')
plt.xlabel('التوقعات الفعلية')
plt.ylabel('التوقعات القابلة للتنبؤ')
plt.show()


