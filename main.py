import keras
import numpy as np
import modelbuilder
import matplotlib.pyplot as plt
import os
import sys

model_save_location = 'model'
fantasy_data_dir = os.getcwd() + "/Fantasy-Premier-League/data"
if 1 in sys.argv and sys.argv[1] == 'build':
    X, y = modelbuilder.build_training_data(fantasy_data_dir)

    training_X = X[100:]
    training_y = y[100:]

    testing_X = X[:100]
    testing_y = y[:100]

    model = modelbuilder.build_model(X, y)
    model.save(model_save_location)

    _, accuracy = model.evaluate(training_X, training_y)
    print("Accuracy: %.2f" % (accuracy * 100))
else:
    model = keras.models.load_model(model_save_location)


X_21, y_21 = modelbuilder.build_year_data(lookback=5, year_path=os.path.join(fantasy_data_dir, "2021-22"))

X_21 = np.asarray(X_21)
y_21 = np.asarray(y_21)

pred_21 = model.predict(X_21)

line_x = np.linspace(-5, 5, 100)
line_y = 2 * line_x + 1

plt.plot(y_21, pred_21, "r.")
plt.plot(line_x, line_y, '-b', label='y=2x+1')
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
