import cv2
import numpy as np
from collections import defaultdict, deque
import os
import csv
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
    od = ObjectDetection("dnn_model/yolov8x.pt")
    od.load_class_names("dnn_model/classes.txt")
    return od

def warpFrame(frame, transformer, target_width, target_height):
    target_size = (int(target_width), int(target_height))
    return cv2.warpPerspective(frame, transformer.m, target_size)

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

def start_tracking(coordinates, real_life_coords, video_path, detection_area, additional_areas, additional_area_names,
                   img_size=1280, confidence=0.5):
    SOURCE = np.array(coordinates)

    TARGET = np.array(
        [
            real_life_coords[0],  # Top-left corner (no adjustment needed)
            [real_life_coords[1][0] - 1, real_life_coords[1][1]],  # Top-right corner
            [real_life_coords[2][0] - 1, real_life_coords[2][1] - 1],  # Bottom-right corner
            [real_life_coords[3][0], real_life_coords[3][1] - 1]  # Bottom-left corner
        ]
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    cap, fps = getVideoInfo(video_path)
    od = LoadObjectDetection()
    tracker = LoadTracker()

    speed_by_id = {}
    alpha = 0.1
    ema_coords = {}

    coordinates_by_id = defaultdict(lambda: deque(maxlen=int(fps)))

    # Data for additional areas
    area_intersections = {i: set() for i in range(len(additional_areas))}

    def create_unique_csv_filename(base_path):
        counter = 1
        base_name = os.path.splitext(base_path)[0]
        extension = os.path.splitext(base_path)[1]

        unique_path = base_path
        while os.path.exists(unique_path):
            unique_path = f"{base_name}({counter}){extension}"
            counter += 1

        return unique_path

    csv_file = create_unique_csv_filename(os.path.splitext(video_path)[0] + "_vehicle_data.csv")

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Area', 'Time', 'Vehicle ID', 'Vehicle Type', 'Speed'])

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Warp the frame for perspective view
            warped_frame = warpFrame(frame, view_transformer, TARGET[1][0] + 1, TARGET[2][1] + 1)
            cv2.namedWindow('warped_frame', cv2.WINDOW_NORMAL)
            cv2.imshow("warped_frame", warped_frame)

            bboxes, class_ids, scores = od.detect(frame, imgsz=img_size, conf=confidence, classes=[2, 3, 5, 7])
            bboxes_ids = tracker.update(bboxes, scores, class_ids, frame)

            for bbox_id in bboxes_ids:
                (x, y, x2, y2, object_id, class_id, score) = np.array(bbox_id)
                cx = int((x + x2) / 2)
                cy = int((y + y2) / 2)  # Use the center of the bbox

                if object_id not in ema_coords:
                    ema_coords[object_id] = (cx, cy)
                else:
                    prev_cx, prev_cy = ema_coords[object_id]
                    cx = alpha * cx + (1 - alpha) * prev_cx
                    cy = alpha * cy + (1 - alpha) * prev_cy
                    ema_coords[object_id] = (cx, cy)

                # Check if inside the perspective transform area
                is_inside_perspective = cv2.pointPolygonTest(SOURCE, (cx, cy), False)
                if is_inside_perspective <= 0:
                    continue

                # Check if inside the detection area
                is_inside_detection = cv2.pointPolygonTest(np.array(detection_area), (cx, cy), False)
                if is_inside_detection <= 0:
                    continue

                color = get_color_for_tracker_id(object_id)

                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

                class_name = od.classes[class_id]
                label = "{} {:.2f} km/h".format(class_name, speed_by_id.get(object_id, 0))
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)

                if object_id not in coordinates_by_id:
                    coordinates_by_id[object_id] = deque(maxlen=int(fps))
                else:
                    coordinates_by_id[object_id].append((cx, cy))

                if len(coordinates_by_id[object_id]) > fps / 3:
                    start_point = coordinates_by_id[object_id][0]
                    end_point = coordinates_by_id[object_id][-1]
                    transformed_start = view_transformer.transform_points(np.array([start_point]))[0]
                    transformed_end = view_transformer.transform_points(np.array([end_point]))[0]
                    distance = np.linalg.norm(transformed_end - transformed_start)
                    time_elapsed = len(coordinates_by_id[object_id]) / fps
                    speed = (distance / time_elapsed) * 3.6
                    speed_by_id[object_id] = speed
                    cv2.rectangle(frame, (x, y), (x + w, y - h - 10), color, -1)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                # Process each additional area
                for i, area in enumerate(additional_areas):
                    area_points = np.array(area)
                    if cv2.pointPolygonTest(area_points, (cx, cy), False) >= 0 and object_id not in area_intersections[i]:
                        area_intersections[i].add(object_id)
                        writer.writerow([additional_area_names[i], frame_count / fps, object_id, class_name,
                                         round(speed_by_id.get(object_id, 0), 2)])

            # Draw detection and additional areas
            cv2.polylines(frame, [np.array(detection_area)], True, (0, 0, 225), 4)
            for area in additional_areas:
                cv2.polylines(frame, [np.array(area)], True, (255, 0, 0), 3)

            # Show the original frame with tracking
            cv2.namedWindow('tracker_frame', cv2.WINDOW_NORMAL)
            cv2.imshow("tracker_frame", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
