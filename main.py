from keras.models import load_model
import numpy as np
import kerasmodel
import dataextraction
import matplotlib.pyplot as plt
import os
import sys

model_save_location = 'model'
fantasy_data_dir = os.getcwd() + "/Fantasy-Premier-League/data"

if len(sys.argv) >= 2 and sys.argv[1] == 'build':
    X, y = dataextraction.build_training_data(fantasy_data_dir, ["2019-20", "2020-21"])

    training_X = X
    training_y = y

    testing_X = X[:100]
    testing_y = y[:100]

    model = kerasmodel.build_model(X, y)
    model.save(model_save_location)

    _, accuracy = model.evaluate(training_X, training_y)
    print("Accuracy: %.2f" % (accuracy * 100))
else:
    model = load_model(model_save_location)

# [6x last year, 5x look back { diff, home, points, minutes }]
year_path = os.path.join(fantasy_data_dir, "2021-22")
X_21, y_21, metadata = dataextraction.build_year_data(lookback=5, year_path=year_path)
player_id_map = dataextraction.build_player_id_map(year_path)
fixture_id_map = dataextraction.build_fixture_id_map(year_path)
team_id_map = dataextraction.build_team_id_map(year_path)

X_21 = np.asarray(X_21)
y_21 = np.asarray(y_21)

last_five_scores = [sum(scores) / len(scores) for scores in X_21[:, 8:-1:4]]
actual_scores = y_21[:, 0]

plt.plot(last_five_scores, actual_scores, "g.")
# plt.plot(actual_minutes, predicted_minutes, "r.")
plt.xlabel("actual")
plt.ylabel("avg last five")
plt.show()



pred_21 = model.predict(X_21)

for i in range(len(pred_21)):
    fixture = fixture_id_map[metadata[i]["fixture"]]
    home = metadata[i]["was_home"]
    if home:
        opponent = team_id_map[fixture["away_team_id"]]
    else:
        opponent = team_id_map[fixture["home_team_id"]]

    player = player_id_map[metadata[i]["player_id"]]
    prediction = round(pred_21[i][0], 2)
    print(f"predicted {prediction} for player '{player['name']}' {'home' if home else 'away'} against {opponent['name']}")

line_x = np.linspace(-5, 30, 2)
line_y = 2 * line_x

actual_scores = y_21[:, 0]
predicted_scores = pred_21[:, 0]

# actual_minutes = y_21[:, 1]
# predicted_minutes = pred_21[:, 1]

plt.plot(actual_scores, predicted_scores, "g.")
# plt.plot(actual_minutes, predicted_minutes, "r.")
plt.plot(line_x, line_y, '-b', label='y=2x+1')
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
