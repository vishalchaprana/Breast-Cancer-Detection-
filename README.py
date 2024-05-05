# Breast-Cancer-Detection-

import pandas as pd
import numpy as np
import plotly.express as px 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC

data = pd.read_csv(r'C:\Users\Vishal Chaprana\OneDrive\Desktop\breast_cancer_survival.csv')
print(data.head())

print(data.isnull().sum())
dat=data.dropna()
data.info()

print(data.Gender.value_counts())

stage=data["Tumour_Stage"].value_counts()
transactions=stage.index
quantity=stage.values

figure=px.pie(data,values=quantity,names=transactions,hole=0.7,title="TUMOUT STAGES OF PATIENTS")
figure.show()

histology = data["Histology"].value_counts()
transactions=histology.index
quantity=histology.values
figure=px.pie (data, values=quantity,names=transactions, hole = 0.5,title="Histology of Patients")
figure.show()

#ER status
print(data["ER status"].value_counts())
#PR status
print(data["PR status"].value_counts())
#HER2 status
print(data["HER2 status"].value_counts())

surgery=data["Surgery_type"].value_counts()
transactions = surgery.index
quantity= surgery.values
figure= px.pie (data, values=quantity, names=transactions, hole=0.6, title="Type of surgery done on patients")
figure.show()

data["Tumour_Stage"] = data["Tumour Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma":3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"]= data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, "Lumpectomy": 3, "Simple Mastectomy": 4})
print(data.head())

x = np.array(data[['Age', 'Gender', 'Proteini', 'Protein2', 'Protein3', 'Protein4', 'Tumour Stage', 'Histology', 'ER status','PR status', 'HER2 status', 'Surgery_type']])
y= np.array(data[['Patient_Status']])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.10, random_state=42)

model=SVC()
model.fit(xtrain,ytrain)

#Prediction
#Features=[['Age', 'Gender', 'Proteini', 'Protein2', 'Protein3', 'Protein4', 'Tumour Stage', 'Histology', 'ER status','PR status', 'HER2 status', 'Surgery_type']])

features=np.array([[46.0,1,1,0.87,-0.54715,0.3,4,1,1,1,2,4]])
print(model.predict(features))
