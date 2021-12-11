import os

import numpy as np
import pandas as pd

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


def build_training_data(directory: str, years: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for year in years:
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

    X: list[np.ndarray] = []
    y: list[np.ndarray] = []

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

    return X, y


def build_actual_scores(year_path) -> dict:
    # format: { id, name, position, team, predictions = [{ week, score, chanceOfPlaying, cost }] }

    players = {}

    players_raw_path = os.path.join(year_path, "players_raw.csv")
    players_raw = pd.read_csv(players_raw_path).T

    for i in players_raw:
        player = players_raw[i]
        players[player["id"]] = {
            "name": f"{player['first_name']} {player['second_name']}",
            "team": player["team"],
            "position": player["element_type"]
        }

    players_path = os.path.join(year_path, 'players')
    actuals = {}

    for file in os.listdir(players_path):
        filename = os.fsdecode(file)
        player_path = os.path.join(players_path, filename)
        if os.path.isdir(player_path):
            player_gw_path = os.path.join(player_path, "gw.csv")
            player_datas = pd.read_csv(player_gw_path).T

            for i in player_datas:
                player_data = player_datas[i]
                player_id = player_data["element"]

                if player_id not in actuals:
                    actuals[player_id] = {
                        "id": player_id,
                        "name": players[player_id]["name"],
                        "position": players[player_id]["position"],
                        "team": players[player_id]["team"],
                        "predictions": []
                    }

                actuals[player_id]["predictions"].append(
                    {
                        "week": player_data["round"],
                        "score": player_data["total_points"],
                        "chanceOfPlaying": 1,
                        "cost": player_data["value"]
                    }
                )

    return actuals
