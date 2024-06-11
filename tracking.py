import cv2
import numpy as np
from collections import defaultdict, deque
import os
from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def LoadObjectDetection():
    od = ObjectDetection("dnn_model/yolov8m.pt")
    od.load_class_names("dnn_model/classes.txt")
    return od

def getVideoInfo(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def LoadTracker():
    mot = MultiObjectTracking()
    tracker = mot.ocsort()
    return tracker

def get_color_for_tracker_id(tracker_id):
    np.random.seed(tracker_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def start_tracking(coordinates, real_life_coords, video_path, detection_area, additional_areas):
    SOURCE = np.array(coordinates)

    # Calculate TARGET_WIDTH and TARGET_HEIGHT from real_life_coords
    TARGET_WIDTH = max([coord[0] for coord in real_life_coords])
    TARGET_HEIGHT = max([coord[1] for coord in real_life_coords])

    TARGET = np.array(
        [
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ]
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    cap, fps = getVideoInfo(video_path)
    od = LoadObjectDetection()
    tracker = LoadTracker()

    vehicles_ids1 = set()
    speed_by_id = {}
    alpha = 0.1
    ema_coords = {}

    coordinates_by_id = defaultdict(lambda: deque(maxlen=int(fps)))

    detection_polygon = np.array(detection_area, dtype=np.int32)
    additional_polygons = [np.array(area, dtype=np.int32) for area in additional_areas]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, class_ids, scores = od.detect(frame, imgsz=1280, conf=0.5)
        bboxes_ids = tracker.update(bboxes, scores, class_ids, frame)

        for bbox_id in bboxes_ids:
            (x, y, x2, y2, object_id, class_id, score) = np.array(bbox_id)

            # Calculate the center of the bottom bounding box side
            bottom_center_x = int((x + x2) / 2)
            bottom_center_y = y2

            if object_id not in ema_coords:
                ema_coords[object_id] = (bottom_center_x, bottom_center_y)
            else:
                prev_x, prev_y = ema_coords[object_id]
                bottom_center_x = int(alpha * bottom_center_x + (1 - alpha) * prev_x)
                bottom_center_y = int(alpha * bottom_center_y + (1 - alpha) * prev_y)
                ema_coords[object_id] = (bottom_center_x, bottom_center_y)

            is_inside_detection = cv2.pointPolygonTest(detection_polygon, (bottom_center_x, bottom_center_y), False)
            is_inside_perspective = cv2.pointPolygonTest(SOURCE, (bottom_center_x, bottom_center_y), False)

            if is_inside_detection > 0 and is_inside_perspective > 0:
                color = get_color_for_tracker_id(object_id)

                cv2.rectangle(frame, (x, y), (x2, y2), color, 1)

                class_name = od.classes[class_id]
                label = "{} {:.2f} km/h".format(class_name, speed_by_id.get(object_id, 0))
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3, 2)

                cv2.rectangle(frame, (x, y), (x + w, y - h - 10), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

                if object_id not in coordinates_by_id:
                    coordinates_by_id[object_id] = deque(maxlen=int(fps))
                else:
                    coordinates_by_id[object_id].append((bottom_center_x, bottom_center_y))

                    if len(coordinates_by_id[object_id]) > fps * (3 / 4):
                        start_point = coordinates_by_id[object_id][0]
                        end_point = coordinates_by_id[object_id][-1]
                        transformed_start = view_transformer.transform_points(np.array([start_point]))[0]
                        transformed_end = view_transformer.transform_points(np.array([end_point]))[0]
                        distance = np.linalg.norm(transformed_end - transformed_start)
                        time_elapsed = len(coordinates_by_id[object_id]) / fps
                        speed = (distance / time_elapsed) * 3.6
                        speed_by_id[object_id] = speed

                vehicles_ids1.add(object_id)

        cv2.putText(frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids1)), (600, 50), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (15, 225, 215), 2)

        cv2.polylines(frame, [detection_polygon], True, (0, 0, 225), 4)

        for area in additional_polygons:
            cv2.polylines(frame, [area], True, (0, 255, 0), 2)

        cv2.namedWindow('tracker_frame', cv2.WINDOW_NORMAL)
        cv2.imshow("tracker_frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
