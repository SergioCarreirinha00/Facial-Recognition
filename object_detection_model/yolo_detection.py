from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import Detection
import cv2

model = YOLO("../best.pt")
model.predict(source="1", show=True, conf=0.5)