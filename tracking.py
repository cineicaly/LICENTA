from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np
import os
import time
from collections import defaultdict, deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Load Object Detection
od = ObjectDetection("dnn_model/yolov8x.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Videocapture
cap = cv2.VideoCapture("vehicles.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. Load tracker
mot = MultiObjectTracking()
tracker = mot.ocsort()

# Object Counting Area
crossing_area_1 = np.array([(1252, 787), (2298, 803), (5039, 2159), (-550, 2159)])

vehicles_ids1 = set()
speed_by_id = {}

# Dictionary to store coordinates and times for each vehicle
coordinates_by_id = defaultdict(lambda: deque(maxlen=int(fps)))

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

# Function to generate a color based on the tracker_id
def get_color_for_tracker_id(tracker_id):
    np.random.seed(tracker_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

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

        is_inside1 = cv2.pointPolygonTest(crossing_area_1, (cx, cy), False)
        # Only process if the object is inside the crossing area
        if is_inside1 <= 0:
            continue

        # Get color for the tracker_id
        color = get_color_for_tracker_id(object_id)

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
        # cv2.circle(frame, (cx, cy), 5, color, -1)

        # Object id and speed display
        label = "{} {:.2f} km/h".format(object_id, speed_by_id.get(object_id, 0))
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 5, 3)

        cv2.rectangle(frame, (x, y), (x + w, y - h - 10), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

        if object_id not in coordinates_by_id:
            coordinates_by_id[object_id] = deque(maxlen=int(fps))
        else:
            coordinates_by_id[object_id].append((cx, cy))

            # Calculate speed using the coordinates stored over one second
            if len(coordinates_by_id[object_id]) > fps / 2:  # Ensure at least half a second of data
                start_point = coordinates_by_id[object_id][0]
                end_point = coordinates_by_id[object_id][-1]
                transformed_start = view_transformer.transform_points(np.array([start_point]))[0]
                transformed_end = view_transformer.transform_points(np.array([end_point]))[0]
                distance = np.linalg.norm(transformed_end - transformed_start)
                time_elapsed = len(coordinates_by_id[object_id]) / fps
                speed = (distance / time_elapsed) * 3.6  # Convert m/s to km/h
                speed_by_id[object_id] = speed

        vehicles_ids1.add(object_id)

    cv2.putText(frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids1)), (600, 50), cv2.FONT_HERSHEY_PLAIN,
                1.5, (15, 225, 215), 2)

    # Draw area
    #cv2.polylines(frame, [crossing_area_1], True, (0, 0, 225), 4)

    # Resize the frame to make the window smaller
    frame_resized = cv2.resize(frame, (1600, 900))

    cv2.imshow("Frame", frame_resized)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()