import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

model = load_model("../face_recognition_100epoch_test.h5")

frame = cv2.imread("img.jpg")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for x, y, w, h in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:

        print("Face not detected")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey: ey + eh, ex:ex + ew]


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

final_image = cv2.resize(face_roi, (48, 48)) # resizing
final_image = np.array(cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY))
final_image = np.expand_dims(final_image, axis = 0) #adding 4 dimension
final_image = final_image/255.0 #normalize


prediction = model.predict(final_image)


print()
