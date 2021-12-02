import numpy as np

import os

from keras.models import Sequential
from keras.layers import Dense

directory = "./Fantasy-Premier-League/data/"
headers = [
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "selected",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards"
]

# what part of the next game do I _know_ to help me predict?
# TODO: include difficulty of fixture, avg. goals conceded/scored per team, position
context_headers = [
    # "fixture",
    # "opponent_team",
    # "value",
]

# what important parts of completed games do I care about?
history_headers = context_headers + [
    "total_points",
    "minutes",
]


# what do I want to predict?
prediction_headers = [
    "total_points",
    # "minutes",
]

lookback = 5

included = [*range(0, 11), *range(12, 29), 30]
X = []
y = []
for year in ["2019-20", "2020-21", "2021-22"]:
    year_path = os.path.join(directory, year, "players")
    if not os.path.isdir(year_path):
        continue

    for file in os.listdir(year_path):
        filename = os.fsdecode(file)
        player_path = os.path.join(year_path, filename)
        if os.path.isdir(player_path):
            player_data = np.genfromtxt(
                os.path.join(player_path, "gw.csv"),
                delimiter=",",
                names=True
            )

            if len(player_data.shape) == 0 or len(player_data) < lookback + 1:
                continue

            last_three_years = [0 for i in range(3 * len(history_headers))]
            history_path = os.path.join(player_path, "history.csv")
            if os.path.isfile(history_path):
                history_data = np.genfromtxt(
                    os.path.join(player_path, "history.csv"),
                    delimiter=",",
                    names=True
                )

                if len(history_data.shape) == 0:
                    last_three_years[1] = history_data["total_points"]
                else:
                    for i in range(min(3, len(history_data))):
                        h_idx = (-i - 1)
                        last_three_years[i] = history_data[h_idx]["total_points"]
                        last_three_years[i + 3] = history_data[h_idx]["minutes"]

            a = 1
            for prediction_week in range(lookback, len(player_data)):
                known_values = last_three_years.copy()

                for gw in player_data[prediction_week - lookback:prediction_week]:
                    for header in history_headers:
                        known_values.append(gw[header])

                for header in context_headers:
                    known_values.append(player_data[prediction_week][header])

                prediction_values = []
                for header in prediction_headers:
                    prediction_values.append(player_data[prediction_week][header])

                X.append(np.asarray(known_values, dtype=float))
                y.append(np.asarray(prediction_values, dtype=float))

print(len(X))
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

data_size = len(X)
p = np.random.permutation(len(X))

X = X[p]
y = y[p]

testing_size = 30

training_X = X[testing_size:data_size - testing_size]
training_y = y[testing_size:data_size - testing_size]

testing_X = X[:testing_size]
testing_y = y[:testing_size]

validation_X = X[data_size - testing_size:]
validation_y = y[data_size - testing_size:]

# model = Sequential()
# model.add(
#     LSTM(10,
#          activation='relu',
#          input_shape=(3, X.shape[1]))
# )
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
#
# model.fit_generator(training_X, epochs=25, verbose=1)
#
# prediction = model.predict_generator(testing_X)
#
# print(prediction)
# print(testing_y)

# define the keras model
model = Sequential()
model.add(Dense(15, input_dim=X.shape[1], activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(y.shape[1], activation="relu"))

model.compile(loss="mse", metrics=["accuracy"], optimizer="adam")

model.fit(training_X, training_y, epochs=10, batch_size=10, validation_data=(validation_X, validation_y))

_, accuracy = model.evaluate(training_X, training_y)
print("Accuracy: %.2f" % (accuracy * 100))

predictions = model.predict(testing_X)

for i in range(len(testing_y)):
    print(testing_X[i])
    print(f"the model predicted {predictions[i]} ==> actually {testing_y[i]}")
    print()
