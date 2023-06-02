import seaborn as sns
from keras.src.utils import load_img
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def data_analysis(data):
    sns.countplot(data=data, x='label')
    plt.show()


def showImage(data):
    img = Image.open(data['image'][0])
    print(img)
    plt.imshow(img, cmap="gray")
    plt.show()


def showImages(data):
    plt.figure(figsize=(25, 25))
    files = data.iloc[0:25]

    for index, file, label in files.itertuples():
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')

    plt.show()




