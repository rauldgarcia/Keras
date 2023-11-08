from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Getting the data ready
# Generate train dummy data for 1000 students and dummy test for 500
# Columns: Age, Hours of study & avg previous test scores
np.random.seed(2018) # Setting seed for reproducibility
train_data, test_data = np.random.random((1000, 3)), np.random.random((500, 3))
# Generate dummy results for 1000 students: whether passed (1) or failed (0)
labels = np.random.randint(2, size=(1000, 1))

# Defining the model structure with the required layers, # of neurons, activation function and optimizer
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and make predictios
model.fit(train_data, labels, epochs=10, batch_size=32)
# Make predictions from the trained model
predictions = model.predict(test_data)

print(test_data)
print(predictions)