import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27: # key 27 = escape key
        break

cam.release()
cv2.destroyAllWindows()