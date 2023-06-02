import random

import matplotlib.pyplot as plt


def test_images(model, test, x_test, le):
    image_index = random.randint(0, len(test))
    print("Original Output:", test['label'][image_index])
    pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
    prediction_label = le.inverse_transform([pred.argmax()])[0]
    print("Predicted Output: ", prediction_label)
    plt.imshow(x_test[image_index].reshape(48,48), cmap= 'gray')
    plt.show()
