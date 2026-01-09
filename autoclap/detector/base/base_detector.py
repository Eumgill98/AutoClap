from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator

import torch

class BaseDetector(BaseModel, ABC):
    """A abstract class for Detector"""
    weight_path: str = Field(..., description="weight file path")
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model: Optional[Any] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def init_model(self):
        raise NotImplementedError(
            f"""
{self.__class__.__name__} must implement `init_model` as follows:

@model_validator(mode="after")
def init_model(self):
    try:
        self.model = [YOUR_MODEL_CLASS](self.weight_path)
        self.model.to(self.device)
    except Exception as e:
        raise ValueError(f"Failed to initialize [YOUR_MODEL_CLASS] model: {{e}}")
    return self
"""
        )

    def to(
        self,
        device: str
    ) -> BaseDetector:
        """
        Move the model to the specified device.

        Args:
            device (str): Target device ('cpu', 'cuda' ...)
        
        Returns:
            self: Returns self for method chaining
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    @abstractmethod
    def predict(
        self,
        image: List[Any],
        **kwargs,
    ):
        ...

    @abstractmethod
    def structure_output(
        self,
        output: List[Any]
    ):
        ...

    @abstractmethod
    def __call__(
        self,
        inputs: Union[Any, List[Any]],
        **kwargs,
    ):
        ...