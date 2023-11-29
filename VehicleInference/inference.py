from ultralytics import YOLO
import os
import cv2


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
    def __init__(self, path_classes: str, conf: float = 0.75, iou: float = 0.75,
                 half: bool = True, max_det: int = 2, imgsz: int = 320):
        self.path_classes = path_classes
        self.conf = conf
        self.iou = iou
        self.half = half
        self.max_det = max_det
        self.imgsz = imgsz
        self.model_factory = YOLOModelFactory()
        self.model = None
        self.images = []

    def _list_dir(self, directory: str) -> list:
        for root, dirs, files in os.walk(directory):
            for file in files:
                abs_path = os.path.join(root, file)
                im = cv2.imread(abs_path)
                im_sized = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
                self.images.append(im_sized)
        return self.images

    def initialize_model(self, path_model: str):
        self.model = self.model_factory.create_model(path_model)

    def inference(self, source: str = None):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")

        self._list_dir(source)
        results = self.model(self.images)

        for result in results:
            boxes = result.boxes
            probs = result.probs

            print(result)
            print(boxes, probs)


# Example usage:
path_to_model = "path/to/your/model"
path_to_classes = "path/to/your/classes"
vehicle_inference = VehicleInference(path_classes=path_to_classes)
vehicle_inference.initialize_model(path_model=path_to_model)
vehicle_inference.inference(source="path/to/your/images")
