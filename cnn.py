import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Loading data from Keras datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Defining the height and weight and number of samples
# Each Image is a 28x28 with 1 channel matrix
training_samples, height, width = x_train.shape
testing_samples, _, _ = x_test.shape

print('Training Samples: ', training_samples)
print('Testing Samples: ', testing_samples)
print('Height: ' + str(height) + ' X Width: ' + str(width))

# Lets have a look at a sample image in the training data
plt.imshow(x_train[0], cmap='gray', interpolation='none')

# We now have to engineer the image data into the right form
# For CNN, we would need the data in Height x Width x Channels
# form since the image is in grayscale, we will use channel = 1
channel = 1
x_train = x_train.reshape(training_samples, height, width, channel).astype('float32')
x_test = x_test.reshape(testing_samples, height, width, channel).astype('float32')

# To improve the training process, we would need to standardize  or normalize the values 
# We can achive this using a simple divive by 256 for all values
x_train = x_train / 255
x_test = x_test / 255

# Total number of digits = 10
target_classes = 10
n_classes = 10

# Convert integer labels into one-hot vectors
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

# Designing the CNN Model
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=(height, width, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)