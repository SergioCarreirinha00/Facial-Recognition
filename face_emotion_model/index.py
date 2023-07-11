import matplotlib.pyplot as plt
import numpy as np
import warnings

from keras.src.utils import to_categorical
from tqdm.notebook import tqdm
from face_emotion_model.preprocessing_data import test_train_data
from face_emotion_model.exploratory_data_analysis import data_analysis, showImage, showImages
from face_emotion_model.feature_extraction import extract_features
from sklearn.preprocessing import LabelEncoder
from face_emotion_model.model_creation import model_creation, save
from face_emotion_model.train_model import train_model
from face_emotion_model.results import drawPlotResultAccuracy
from keras.models import load_model
from face_emotion_model.hardtest import test_images

warnings.filterwarnings('ignore')


#
train, test = test_train_data()

data_analysis(train)
data_analysis(test)

showImages(train)

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])
#
# ## normalize the image
x_train = train_features / 255.0
x_test = test_features / 255.0
#
#
## convert label to integer
le = LabelEncoder()
le.fit(train['label'])

y_train= le.transform(train['label'])
y_test = le.transform(test['label'])

print(y_train[1], le.inverse_transform([y_train[1]]))
#
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

#
model = model_creation()

history = train_model(model, x_train, y_train, x_test, y_test)

save(model)



drawPlotResultAccuracy(history)

model = load_model('face_recognition_1epoch_test.h5')
#
# test_images.jpg(model, test, x_test, le)



