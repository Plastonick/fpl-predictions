from keras.models import Sequential
from keras.layers import Dense


def build_model(X, y):
    training_X = X[30:]
    training_y = y[30:]

    validation_X = X[:30]
    validation_y = y[:30]

    # define the keras model
    model = Sequential()
    model.add(Dense(6, input_dim=X.shape[1], kernel_initializer="uniform", activation="relu"))
    model.add(Dense(6, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(6, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(y.shape[1], kernel_initializer="uniform", activation="sigmoid"))

    model.compile(loss="mae", metrics=["accuracy"], optimizer="adam")

    model.fit(training_X, training_y, epochs=5, batch_size=10, validation_data=(validation_X, validation_y))

    return model
