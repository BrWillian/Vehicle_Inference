from ultralytics import YOLO
import ultralytics
import os
import numpy as np
import cv2
import json
from typing import Dict


class YOLOModelFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModelFactory, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model not initialized. Call create_model first.")
        return self._model

    def create_model(self, path_model):
        self._model = YOLO(path_model)
        return self._model


class VehicleInference:
    def __init__(self, classes: dict = dict(), conf: float = 0.75, iou: float = 0.75,
                 half: bool = True, max_det: int = 2, imgsz: int = 640):
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.half = half
        self.max_det = max_det
        self.imgsz = imgsz
        self.model_factory = YOLOModelFactory()
        self.model = None
        self.images = []
        self.images_name = []

    def _list_dir(self, directory: str) -> list:
        for root, dirs, files in os.walk(directory):
            for file in files:
                abs_path = os.path.join(root, file)
                im = cv2.imread(abs_path)
                self.images_name.append(abs_path.split(".")[0])
                self.images.append(im)
        return self.images

    def initialize_model(self, path_model: str):
        self.model = self.model_factory.create_model(path_model)
        self.classes = self.model.names

    @staticmethod
    def get_max_confidence_from_json(json_result: str):
        result_json = json.loads(json_result)
        index = max(range(len(result_json)), key=lambda index: result_json[index]['confidence'])
        return result_json[index]

    def inference(self, source: str = None, output: str = None):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")

        self._list_dir(source)

        for index, image in enumerate(self.images):
            image_copy = image.copy()
            results = self.model(image.copy(), half=True)
            for result in results:
                result_by_confidence = self.get_max_confidence_from_json(result.tojson())
                vehicle_class = result_by_confidence["name"]
                box = result_by_confidence["box"]

                if vehicle_class == "carro" or vehicle_class == "caminhonete":
                    roi = image_copy[int(box['y1']):int(box['y2']), int(box['x1']):int(box['x2'])]
                    cv2.imwrite(self.images_name[index]+"_roi.jpg", roi)
                else:
                    print(self.images_name[index])
