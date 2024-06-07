import cv2
import numpy as np

from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getVideoInfo(path):
    path = '/demo/vehicles.mp4'

# 1. Load Object Detection
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Videocapture
cap = cv2.VideoCapture("demo/cars_highway.mp4")

# 3. Load tracker
mot = MultiObjectTracking()
tracker = mot.ocsort()

# Object Counting Area
crossing_area_1 = np.array([(617, 393), (964, 389), (926, 498), (477, 485)])
crossing_area_2 = np.array([(1179, 349), (1275, 341), (1275, 402), (1155, 438)])
vehicles_ids1 = set()
vehicles_ids2 = set()
trajectory_by_id = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. We detect the objects
    bboxes, class_ids, scores = od.detect(frame, imgsz=640, conf=0.5)

    # 2. We track the objects detected
    bboxes_ids = tracker.update(bboxes, scores, class_ids, frame)
    for bbox_id in bboxes_ids:
        (x, y, x2, y2, object_id, class_id, score) = np.array(bbox_id)
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        # draw bounding box
        # cv2.rectangle(frame, (x, y), (x2, y2), od.colors[class_id], 2)
        cv2.circle(frame, (cx, cy), 5, od.colors[class_id], -1)

        # Object id
        cv2.putText(frame, "{}".format(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.4, od.colors[class_id], 2)

        if object_id not in trajectory_by_id:
            trajectory_by_id[object_id] = [(cx, cy)]
        else:
            trajectory_by_id[object_id].append((cx, cy))

            # Draw Trajectory
            trajectory = trajectory_by_id[object_id]
            cv2.polylines(frame, [np.array(trajectory[-20:])], False, (15, 225, 215), 2)

        is_inside1 = cv2.pointPolygonTest(crossing_area_1, (cx, cy), False)
        is_inside2 = cv2.pointPolygonTest(crossing_area_2, (cx, cy), False)
        # If the object is inside the crossing area
        if is_inside1 > 0:
            vehicles_ids1.add(object_id)
        elif is_inside2 > 0:
            vehicles_ids2.add(object_id)



    cv2.putText(frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids1)), (600, 50), cv2.FONT_HERSHEY_PLAIN,
                1.5, (15, 225, 215), 2)
    cv2.putText(frame, "VEHICLES AREA 2: {}".format(len(vehicles_ids2)), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                1.5, (15, 225, 215), 2)

    # Draw area
    cv2.polylines(frame, [crossing_area_1], True, (15, 225, 215), 2)
    cv2.polylines(frame, [crossing_area_2], True, (15, 225, 215), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()