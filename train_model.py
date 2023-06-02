def train_model(model, x_train, y_train, x_test, y_test):
    # train the model
    history = model.fit(x=x_train, y=y_train, batch_size=129, epochs=100, validation_data=(x_test, y_test))
    return history
