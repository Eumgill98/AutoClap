from ultralytics import YOLO
from ultralytics.engine.results import Results

from pydantic import model_validator
from typing import Dict, List, Any, Union

from autoclap.detector.base import BaseDetector

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
        image: List[Any],
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
                source=image,
                **kwargs
            )
        )

    def structure_output(
        self,
        output: List[Results]
    ) -> List[Dict]:
        """
        YOLOv8 output to structure.

        Args:
            output (List[Results]): YOLOv8 model predict outputs.

        Returns:
            List of Dict that is structure.
            Each dict contains:
                - boxes: List of bounding boxes in xyxy format
                - scores: List of confidence scores
                - classes: List of class indices
                - class_names: List of class names
        """
        structured_results = []

        for result in output:
            structured_results.append({
                "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                "scores": result.boxes.conf.cpu().numpy().tolist(),
                "classes": result.boxes.cls.cpu().numpy().tolist(),
                "class_names": [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
            })

        return structured_results
    
    def __call__(
        self,
        inputs: Union[Any, List[Any]],
        **kwargs,
    ):
        """
        Perform prediction and return structured output.
        This method connects model.predict with structure_output.

        Args:
            iputs: Single image or List of images for prediction
        
        Returns:
            List of Dict containing structured detection information.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        results = self.predict(inputs, **kwargs)
        return self.structure_output(results)
