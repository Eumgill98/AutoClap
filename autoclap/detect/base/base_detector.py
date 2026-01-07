from abc import ABC, abstractmethod
from typing import Any, Iterable
from pydantic import BaseModel, Field, ConfigDict

import torch

from autoclap.core.sampler import BaseVideoSampler

class BaseDetector(BaseModel, ABC):
    """A abstract class for Detector"""
    weight_path: str = Field(..., description="weight file path")
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @abstractmethod
    def predict(
        self,
        image: Any,
        **kwargs,
    ):
        ...

    @abstractmethod
    def predict_video(
        self,
        video: Any,
        video_sampler: BaseVideoSampler,
        **kwargs,
    ) -> Iterable[Any]:
        ...
