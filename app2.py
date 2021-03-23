import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

df = pd.read_csv("iris.csv") #you can download csv and create data frame 

from sklearn import preprocessing
sp = preprocessing.LabelEncoder()
df['species'] = sp.fit_transform(df['species'])
df['species'].unique()

x=df.iloc[:,0:4].values
y=df.iloc[:,4].values

#now splitng data into training and testing of data
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(train_x,train_y)

pickle.dump(lg, open('logistic.pkl', 'wb'))
