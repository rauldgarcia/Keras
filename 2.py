import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_val = x_train[300:,]
y_val = y_train[300:,]

np.random.seed(2018)

# Define the model architecture
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu')) # Layer 1
model.add(Dense(6, kernel_initializer='normal', activation='relu')) # Layer 2
model.add(Dense(1, kernel_initializer='normal')) # Layer 3

# Configure the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val))

# Evaluate the model
results = model.evaluate(x_test, y_test)

for i in range(len(model.metrics_names)):
    print(model.metrics_names[i], " : ", results[i])

