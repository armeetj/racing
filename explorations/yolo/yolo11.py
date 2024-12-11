"""
YOLOv11 intro demo
- Written by @armeetj
"""

from ultralytics import YOLO
import cv2
import time
import hashlib
import os
import numpy as np

GPU = os.environ.get("GPU", True)
DEBUG = os.environ.get("DEBUG", False)


print("Running demo")
print("GPU:", GPU)
print("DEBUG:", DEBUG)

model = YOLO("weights/yolo11x-seg.pt", verbose=False)
if GPU:
    model.to("cuda")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

fps = 0
prev_t = time.time()


def string_to_color(name):
    hash_object = hashlib.md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


while True:
    ret, img = cap.read()

    # Calculate the FPS
    curr_t = time.time()
    fps = 1 / (curr_t - prev_t)
    prev_t = curr_t

    if cv2.waitKey(1) == ord("q"):
        break

    results = model(img, verbose=False)
    res = results[0]
    for idx, res_dict in enumerate(res.summary()):
        name = res_dict["name"]
        color = string_to_color(name)

        # draw mask
        x, y = res_dict["segments"]["x"], res_dict["segments"]["y"]
        points = np.array([list(zip(x, y))], dtype=np.int32)
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.polylines(img, points, isClosed=True, color=color, thickness=2)
        cv2.fillPoly(mask, points, color)

        # add text
        text_position = (int(x[0]), int(y[0]) - 10)
        cv2.putText(img, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        img = cv2.addWeighted(img, 0.8, mask, 0.2, 0)

        if DEBUG:
            print(idx, name)
            print(len(x))

    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )
    cv2.imshow("Webcam", img)

cap.release()
cv2.destroyAllWindows()
