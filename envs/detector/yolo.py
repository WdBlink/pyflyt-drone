from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEPESegTrainer, YOLOEVPTrainer
import supervision as sv
from dataclasses import dataclass
import torch
import numpy as np
from ultralytics import YOLO


@dataclass
class YoloConfig:
    weight : str
    names : []
    device : torch.device
    refer_image : []  # Reference image used to get visual prompts
    visual_prompts : np.array([])
    predictor : YOLOEVPSegPredictor

@dataclass
class Yolo11Config:
    weight : str
    device : torch.device
    data : str

@dataclass
class YoloeTrainConfig:
    weight : str
    device : torch.device
    data_config : str

class YoloeDetector:
    def __init__(self, config: YoloConfig):
        self.model = YOLOE(config.weight)
        # self.model.fuse()
        self.model.to(config.device)
        # self.model.set_classes(config.names, self.model.get_text_pe(config.names))
        self.config = config

    def detect(self, image):
        # Define visual prompts based on a separate reference image
        # Run prediction on a different image, using reference image to guide what to look for
        self.model.eval()
        results = self.model.predict(
            image,  # Target image for detection
            refer_image=self.config.refer_image,  # Reference image used to get visual prompts
            visual_prompts=self.config.visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        # results = self.model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]
        return labels, detections

    def train(self):
        head_index = len(self.model.model.model) - 1
        freeze = [str(f) for f in range(0, head_index)]
        for name, child in self.model.model.model[-1].named_children():
            if "cv3" not in name:
                freeze.append(f"{head_index}.{name}")

        freeze.extend(
            [
                f"{head_index}.cv3.0.0",
                f"{head_index}.cv3.0.1",
                f"{head_index}.cv3.1.0",
                f"{head_index}.cv3.1.1",
                f"{head_index}.cv3.2.0",
                f"{head_index}.cv3.2.1",
            ]
        )

        self.model.train(data=self.config.data_config, batch=128, epochs=2,  close_mosaic=2, \
                    optimizer='AdamW', lr0=16e-3, warmup_bias_lr=0.0, \
                    weight_decay=0.025, momentum=0.9, workers=4, \
                    trainer=YOLOEVPTrainer, device='0', freeze=freeze)


class Yolo11Detector:
    def __init__(self, config: Yolo11Config):
        self.model = YOLO(config.weight)
        # self.model.fuse()
        self.model.to(config.device)
        self.model.eval()
        # self.model.set_classes(config.names, self.model.get_text_pe(config.names))
        self.config = config

    def detect(self, image):
        # Define visual prompts based on a separate reference image
        # Run prediction on a different image, using reference image to guide what to look for
        results = self.model.predict(
            image  # Target image for detection
        )
        # results = self.model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]
        return labels, detections
    def train(self):
        # Train the model with 2 GPUs
        results = self.model.train(data=self.config.data, epochs=100, imgsz=640, device=[0])