from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np
import os
import time  # Import the time module
from collections import defaultdict, deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Load Object Detection
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Videocapture
cap = cv2.VideoCapture("vehicles.mp4")

# 3. Load tracker
mot = MultiObjectTracking()
tracker = mot.ocsort()

# Object Counting Area
crossing_area_1 = np.array([(1252, 787), (2298, 803), (5039, 2159), (-550, 2159)])

vehicles_ids1 = set()

# Deques to store coordinates and times for each vehicle
trajectory_by_id = defaultdict(lambda: deque(maxlen=30))
time_by_id = defaultdict(lambda: deque(maxlen=30))
speed_by_id = {}

# Source and destination points for perspective transform
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

# Define the ViewTransformer class
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Initialize the ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

fps = cap.get(cv2.CAP_PROP_FPS)
min_frames_between_calculations = int(fps / 2)  # Minimum half a second apart

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
        cv2.rectangle(frame, (x, y), (x2, y2), od.colors[class_id], 2)
        # cv2.circle(frame, (cx, cy), 5, od.colors[class_id], -1)

        # Object id
        cv2.putText(frame, "{}".format(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.4, od.colors[class_id], 2)

        if object_id not in trajectory_by_id:
            trajectory_by_id[object_id] = deque(maxlen=30)
            time_by_id[object_id] = deque(maxlen=30)
            speed_by_id[object_id] = 0
        else:
            trajectory_by_id[object_id].append((cx, cy))
            time_by_id[object_id].append(time.time())

            # Calculate Speed only if there are enough frames and they are spaced apart
            if len(trajectory_by_id[object_id]) >= 2:
                time_diff = time_by_id[object_id][-1] - time_by_id[object_id][0]
                if time_diff >= 0.5:  # Only consider calculations if at least half a second has passed
                    transformed_points = view_transformer.transform_points(np.array(trajectory_by_id[object_id]))
                    p1_trans = transformed_points[0]
                    p2_trans = transformed_points[-1]
                    distance = np.linalg.norm(p2_trans - p1_trans)
                    speed = distance / time_diff * 3.6  # Convert m/s to km/h
                    speed_by_id[object_id] = speed

        is_inside1 = cv2.pointPolygonTest(crossing_area_1, (cx, cy), False)
        # If the object is inside the crossing area
        if is_inside1 > 0:
            vehicles_ids1.add(object_id)

        # Display speed
        cv2.putText(frame, "{:.2f} km/h".format(speed_by_id[object_id]), (cx, cy + 20), cv2.FONT_HERSHEY_PLAIN,
                    1.4, (0, 255, 0), 2)

    cv2.putText(frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids1)), (600, 50), cv2.FONT_HERSHEY_PLAIN,
                1.5, (15, 225, 215), 2)

    # Draw area
    cv2.polylines(frame, [crossing_area_1], True, (15, 225, 215), 2)

    # Resize the frame to make the window smaller
    frame_resized = cv2.resize(frame, (960, 540))

    cv2.imshow("Frame", frame_resized)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
