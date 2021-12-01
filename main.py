import numpy as np

# from os.path import isdir, join
import os

from keras.models import Sequential
from keras.layers import Dense

directory = "./Fantasy-Premier-League/data/2021-22/players/"

included = [*range(0, 11), *range(12, 29), 30]
inputdim = 110
X = []
y = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    player_path = os.path.join(directory, filename)
    if os.path.isdir(player_path):
        included = [*range(0, 11), *range(12, 29), 30]

        player_data = np.loadtxt(
            os.path.join(player_path, "gw.csv"),
            delimiter=",",
            skiprows=1,
            usecols=included,
        )

        shape = player_data.shape
        if len(shape) < 2:
            continue

        player_X = np.asarray(player_data[:-1, :10].flatten(), dtype=float)
        if len(player_X) != inputdim:
            continue

        X = [*X, player_X]
        y = [*y, *player_data[-1:, -6]]

X = np.asarray(X)
y = np.asarray(y)
a = 1
# dataset = np.loadtxt(
#     "./data/data/2021-22/players/Aaron_Cresswell_411/gw.csv",
#     delimiter=",",
#     skiprows=1,
#     usecols=[*range(0, 11), *range(12, 29), 30],
# )

# print(dataset.shape)

# dataset2 = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#
# # # X2 = dataset[:-1, :]
# X2 = dataset2[:-1, :]
# y2 = dataset2[-1:, -6]

# load the dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (y) variables
training_X = X[10:]
training_y = y[10:]

predictor_X = X[:10]
predictor_y = y[:10]

print(training_y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=inputdim, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="relu"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(training_X, training_y, epochs=30, batch_size=5)

_, accuracy = model.evaluate(training_X, training_y)
print("Accuracy: %.2f" % (accuracy * 100))

predictions = model.predict(predictor_X)

print(predictor_y)
print(predictions)
