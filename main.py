import keras
import numpy as np
import kerasmodel
import dataextraction
import sqlextraction
import matplotlib.pyplot as plt
import os
import sys

sqlextraction.build_training_data()

model_save_location = 'model'
fantasy_data_dir = os.getcwd() + "/Fantasy-Premier-League/data"

if len(sys.argv) >= 2 and sys.argv[1] == 'build':
    # X2, y2 = dataextraction.build_training_data(directory=fantasy_data_dir, years=["2020-21"])
    X, y = sqlextraction.build_training_data(form_size=5)

    training_X = X
    training_y = y

    testing_X = X[:100]
    testing_y = y[:100]

    model = kerasmodel.build_model(X, y)
    model.save(model_save_location)

    _, accuracy = model.evaluate(training_X, training_y)
    print("Accuracy: %.2f" % (accuracy * 100))
else:
    model = keras.models.load_model(model_save_location)

# [6x last year, 5x look back { diff, home, points, minutes }]
X_21, y_21 = dataextraction.build_year_data(lookback=5, year_path=os.path.join(fantasy_data_dir, "2021-22"))

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

line_x = np.linspace(-5, 30, 2)
line_y = 2 * line_x

actual_scores = y_21[:, 0]
predicted_scores = pred_21[:, 0]

actual_minutes = y_21[:, 1]
predicted_minutes = pred_21[:, 1]

plt.plot(actual_scores, predicted_scores, "g.")
# plt.plot(actual_minutes, predicted_minutes, "r.")
plt.plot(line_x, line_y, '-b', label='y=2x+1')
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
