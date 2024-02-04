from keras import Sequential
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.layers import Dense
import numpy as np

# Generate dummy data for 3 features and 1000 samples
x_train = np.random.random((1000, 3))

# Generate dummy results for 1000 samples: 1 or 0
y_train = np.random.randint(2, size=(1000, 1))

# Create a python function that returns a compiled DNN model
def create_dnn_model():
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Use Keras wrapper to package the model as an sklearn object
model = KerasClassifier(build_fn=create_dnn_model)

# Define the grid search parameters
batch_size = [32, 64, 128]
epochs = [15, 30, 60]

# Create a list with the parameters
param_grid = {'batch_size':batch_size, 'epochs':epochs}

# Invoke the grid search method with the list of hyperparameters
grid_model = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# Train the model
grid_model.fit(x_train, y_train)

# Extract the best model grid search
best_model = grid_model.best_estimator_