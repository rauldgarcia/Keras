import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')

x = iris.iloc[:,:4].values
y = iris['Species']

# LabelEncoder converts categorical variables from str to int
le = LabelEncoder()
a = le.fit_transform(y).reshape(-1, 1)
# OneHotEncoder converts categorical variables to multiple binary categorical variables
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(a)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)


history = History()
model = load_model('dnn_model.h5')
result = model.evaluate(x_test, y_test)

for i in range(len(model.metrics_names)):
    print('Metric ', model.metrics_names[i], ':', str(round(result[i], 2)))
