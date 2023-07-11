import pandas as pd
import os

# load the dataset

TRAIN_DIR = '../facial-expression-dataset/train/train/'
TEST_DIR = '../facial-expression-dataset/test/test/'


def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        for filename in os.listdir(directory + label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)

        print(label, "Completed")

    return image_paths, labels


def test_train_data():
    # convert into dataFrame
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(TRAIN_DIR)
    print(train['label'])
    # shuffle the dataset
    train = train.sample(frac=1).reset_index(drop=True)
    train.head()

    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(TEST_DIR)
    test.head()

    return train, test

