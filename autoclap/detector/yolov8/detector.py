from ultralytics import YOLO
from ultralytics.engine.results import Results

from pydantic import model_validator
from typing import List, Any

from autoclap.detector.base import BaseDetector
from autoclap.core.output import Detection, DetectorOutput

class YOLOv8Detector(BaseDetector):
    @model_validator(mode="after")
    def init_model(self):
        try:
            self.model = YOLO(self.weight_path)
            self.model.to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to initialize YOLO model: {e}")
        return self
    
    def predict(
        self,
        images: List[Any],
        **kwargs,
    ) -> List[Results]:
        """
        Predict objects in an image using YOLOv8.

        Args:
            image (Any): Input image
            **kwargs: Additional arguments passed to YOLO.predict()

        Returns:
            List of Results objects containing detection information
        """
        return (
            self.model.predict(
                source=images,
                **kwargs
            )
        )

    def structure_output(
        self,
        outputs: List[Results]
    ) -> List[DetectorOutput]:
        """
        YOLOv8 output to structure.

        Args:
            output (List[Results]): YOLOv8 model predict outputs.

        Returns:
            List of DetectorOutput that is structure by Detection
            Each Detectoin contains:
                - boxe:  xyxy format
                - score: confidence scores
                - class_id: class indices
                - class_name: class names
        """
        total_result = []

        for result in outputs:
            dets = []
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy().tolist(),
                result.boxes.conf.cpu().numpy().tolist(),
                result.boxes.cls.cpu().numpy().tolist(),
            ):
                det = Detection(
                    bbox=box,
                    score=conf,
                    class_id=cls,
                    class_name=result.names[int(cls)],
                )
                dets.append(det)
            total_result.append(DetectorOutput(detections=dets))

        return total_result