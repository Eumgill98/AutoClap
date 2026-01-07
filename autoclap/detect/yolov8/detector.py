from ultralytics import YOLO
from pydantic import Field, model_validator
from typing import Optional, Any, List

from autoclap.detect.base import BaseDetector
from autoclap.core.sampler import BaseVideoSampler

class YOLOv8Detector(BaseDetector):
    model: Optional[YOLO] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def init_model(self):
        self.model = YOLO(self.weight_path)
        return self
    
    def predict(
        self,
        image: Any,
        conf: float = 0.3,
        iou: float = 0.5,
        **kwargs,
    ):
        return (
            self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                **kwargs
            )
        )
    
    def predict_video(
        self, 
        video: Any, 
        video_sampler: BaseVideoSampler, 
        batch_size: int = 32,
        conf: float = 0.3,
        iou: float = 0.5,
        **kwargs,
    ) -> List[Any]:
        result = []
        frames = video_sampler.sample(video=video)

        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            tmp = self.model.predict(
                source=batch,
                conf=conf,
                iou=iou,
                **kwargs,
            )
            result.extend(tmp)
        return result

