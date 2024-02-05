from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence

# Setting a max cap for the number of distinct words
top_words = 5000

# Loading the training and test data from keras datasets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# Since the length of each text will be varying
# We will pad the sequences (i.e. text) to get a uniform lenght throughout
max_text_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_text_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_text_length)

# Design the network
embedding_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_length, input_length=max_text_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: ', scores[1])