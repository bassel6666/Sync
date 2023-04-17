import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers.legacy import SGD

# Initialize lemmatizer from NLTK library
lemmatizer = WordNetLemmatizer()

# Load intents from json file
with open('Intent.json') as f:
    intents = json.load(f)

# Preprocess training data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ';', ':'] # to ignore punctuation and stopwords from nltk

for intent in intents['intents']:
    for pattern in intent['text']:
        # Tokenize words in each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add pattern and intent to documents list
        documents.append((word_list, intent['intent']))
        # Add intent to classes list if not already there
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

# Lemmatize words and remove ignore_letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Create training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = [1 if word in document[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split training data into input and output
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model with SGD optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=8, verbose=1, shuffle=True)

# Save model to file
model.save('Bassel_ChatBot.h5')

# Print message when done
print('Training complete. Model saved as "Bassel_ChatBot.model".')