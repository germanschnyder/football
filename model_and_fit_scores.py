from pprint import pprint

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

data = []

keras.callbacks.TensorBoard(log_dir='./GraphScores', histogram_freq=0, write_graph=True, write_images=True)

with open('data.json') as f:
    json_arr = json.load(f)

    # Build name array
    name_set = set()
    for match in json_arr:
        for w in match['white']['players']:
            name_set.add(w)
        for b in match['black']['players']:
            name_set.add(b)

    names = list(name_set)

    np.save('name-list', names)

    for match in json_arr:
        row = np.zeros(len(names) + 2, dtype=int)
        for w in match['white']['players']:
            row[names.index(w)] = 1
        for b in match['black']['players']:
            row[names.index(b)] = -1

        row[len(row) - 2] = match['white']['score']
        row[len(row) - 1] = match['black']['score']

        data.append(row)

data = np.array(data)

np.random.shuffle(data)
X = data[:, :-2]
y = data[:, -2:]

pprint(X)
pprint(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = Sequential()

model.add(Dense(14, activation='relu', input_dim=len(names)))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='linear'))


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./GraphScores', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, y_train, epochs=200, batch_size=4, callbacks=[tbCallBack])

model.save('scores-model')

score = model.evaluate(X_test, y_test)

pprint(score)
