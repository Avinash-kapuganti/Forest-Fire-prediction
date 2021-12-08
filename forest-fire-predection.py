#Required libraries
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

#Read File
df=pd.read_csv("Forest_fire.csv")
#independent columns
x=df[["Oxygen","Temperature","Humidity" ]]
y=df["Fire Occurrence"]

#Train the data
#split and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#using LogisticRegression()
regr=LogisticRegression()
regr.fit(X_train, y_train)
#print the accuracy of the model
print(regr.score(X_test,  y_test))

pickle.dump(regr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
