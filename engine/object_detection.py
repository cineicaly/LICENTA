import numpy as np
from ultralytics import YOLO
import random
import colorsys
import torch

# Set random seed
random.seed(2)


class ObjectDetection:
    def __init__(self, weights_path):
        # Load Network
        self.weights_path = weights_path

        self.classes = None
        self.colors = self.random_colors(80)

        # Load Yolo
        self.model = YOLO(self.weights_path)

        # Load Default device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")

    def random_colors(self, N, bright=False):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


    def load_class_names(self, classes_path):
        # Count classes
        class_n = 0
        self.classes = []
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)
                class_n += 1

        self.colors = self.random_colors(80)

    def detect(self, frame, imgsz=640, conf=0.25, nms=0.45, classes=None, device=None):
        # Filter classes
        filter_classes = classes if classes else None
        device = device if device else self.device
        # Detect objects
        results = self.model.predict(source=frame, save=False, save_txt=False,
                                     imgsz=imgsz,
                                     conf=conf,
                                     nms=nms,
                                     classes=filter_classes,
                                     half=False,
                                     device=device)  # save predictions as labels

        # Get the first result from the array as we are only using one image
        result = results[0]
        # Get bboxes
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        # round score to 2 decimal places
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, scores