from keras.models import Sequential
from keras.layers import Dense, Dropout


import numpy as np
import json

from sklearn.model_selection import train_test_split

data = []

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

    for match in json_arr:
        row = np.zeros(len(names) + 1, dtype=int)
        for w in match['white']['players']:
            row[names.index(w)] = 1
        for b in match['black']['players']:
            row[names.index(b)] = -1

        # use diff for now
        diff = abs(match['white']['score'] - match['black']['score'])
        row[len(row)-1] = 1 if diff < 4 else 0
        # row[len(row) - 1] = diff

        data.append(row)

data = np.array(data)

np.random.shuffle(data)
X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = Sequential()

model.add(Dense(2, activation='relu', input_dim=len(names)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=5)

score = model.evaluate(X_test, y_test)