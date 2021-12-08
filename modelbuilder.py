import os

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from numpy import ndarray

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


def build_model(X, y):
    training_X = X[30:]
    training_y = y[30:]

    validation_X = X[:30]
    validation_y = y[:30]

    # define the keras model
    model = Sequential()
    model.add(Dense(6, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(y.shape[1], activation="relu"))

    model.compile(loss="mae", metrics=["accuracy"], optimizer="adam")

    model.fit(training_X, training_y, epochs=5, batch_size=10, validation_data=(validation_X, validation_y))

    return model


def build_training_data(directory) -> tuple[ndarray, ndarray]:
    X = []
    y = []
    for year in ["2019-20", "2020-21"]:
        yr_X, yr_y = build_year_data(year_path=os.path.join(directory, year), lookback=5)

        X = [*X, *yr_X]
        y = [*y, *yr_y]

    X = np.asarray(X)
    y = np.asarray(y)

    # the order of the training data should make no difference... in practicality... ?
    # TODO check if the order matters
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    return X, y


def build_year_data(lookback, year_path):
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
        "minutes",
    ]

    X: list[ndarray] = []
    y: list[ndarray] = []

    players_path = os.path.join(year_path, "players")
    fixtures_path = os.path.join(year_path, "fixtures.csv")
    fixture_difficulty_map = {}
    fixtures = pd.read_csv(fixtures_path).T

    for i in range(fixtures.shape[1]):
        fixture = fixtures[i]
        fixture_difficulty_map[fixture["id"]] = (fixture["team_h_difficulty"], fixture["team_a_difficulty"])

    for file in os.listdir(players_path):
        filename = os.fsdecode(file)
        player_path = os.path.join(players_path, filename)
        if os.path.isdir(player_path):
            player_gw_path = os.path.join(player_path, "gw.csv")
            player_data = pd.read_csv(player_gw_path).T

            if player_data.shape[1] < lookback + 1:
                continue

            last_three_years = [0 for _ in range(6)]
            history_path = os.path.join(player_path, "history.csv")
            if os.path.isfile(history_path):
                history_data = pd.read_csv(history_path).T
                num_histories = history_data.shape[1]

                for i in range(min(3, num_histories)):
                    h_idx = num_histories - i - 1
                    last_three_years[i] = history_data[h_idx]["total_points"]
                    last_three_years[i + 3] = history_data[h_idx]["minutes"]

            kvheaders = []
            for prediction_week in range(lookback, player_data.shape[1]):
                known_values = last_three_years.copy()
                kvheaders = [*kvheaders, *["ly_tp", "ly_tm", "ly_tp", "ly_tm", "ly_tp", "ly_tm"]]

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

    return X, y
