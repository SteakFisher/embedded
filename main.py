import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

img = cv2.imread("plant3.jpg")
results = model(img, conf=0.05)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Plants", img)
cv2.waitKey(0)
