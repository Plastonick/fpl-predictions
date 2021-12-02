import numpy as np
import pandas as pd

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
    # "was_home",
]

# what important parts of completed games do I care about?
look_back_headers = context_headers + [
    "total_points",
    "minutes",
]

# what do I want to predict?
prediction_headers = [
    "total_points",
    # "minutes",
]

lookback = 5

X = []
y = []
for year in ["2019-20", "2020-21", "2021-22"]:
    year_path = os.path.join(directory, year, "players")
    if not os.path.isdir(year_path):
        continue

    fixtures_path = os.path.join(directory, year, "fixtures.csv")
    fixture_difficulty_map = {}
    fixtures = pd.read_csv(fixtures_path).T

    for i in range(fixtures.shape[1]):
        fixture = fixtures[i]
        fixture_difficulty_map[fixture["id"]] = (fixture["team_h_difficulty"], fixture["team_a_difficulty"])

    for file in os.listdir(year_path):
        filename = os.fsdecode(file)
        player_path = os.path.join(year_path, filename)
        if os.path.isdir(player_path):
            player_gw_path = os.path.join(player_path, "gw.csv")
            # player_data = np.genfromtxt(
            #     os.path.join(player_path, "gw.csv"),
            #     delimiter=",",
            #     names=True
            # )
            player_data = pd.read_csv(player_gw_path).T

            if player_data.shape[1] < lookback + 1:
                continue

            last_three_years = [0 for i in range(6)]
            history_path = os.path.join(player_path, "history.csv")
            if os.path.isfile(history_path):
                history_data = pd.read_csv(history_path).T
                num_histories = history_data.shape[1]

                for i in range(min(3, num_histories)):
                    h_idx = num_histories - i - 1
                    last_three_years[i] = history_data[h_idx]["total_points"]
                    last_three_years[i + 3] = history_data[h_idx]["minutes"]

            a = 1
            for prediction_week in range(lookback, player_data.shape[1]):
                known_values = last_three_years.copy()

                for i in range(prediction_week - lookback, prediction_week):
                    gw = player_data[i]
                    fixture_id = gw["fixture"]
                    was_home = gw["was_home"]

                    if was_home:
                        difficulty = fixture_difficulty_map[fixture_id][0]
                    else:
                        difficulty = fixture_difficulty_map[fixture_id][1]

                    known_values.append(difficulty)
                    known_values.append(1 if was_home else 0)

                    for header in look_back_headers:
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
    print([int(n) for n in testing_X[i]])
    print(f"the model predicted {predictions[i]} ==> actually {testing_y[i]}")
    print()
