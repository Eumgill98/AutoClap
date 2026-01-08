from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

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
    def __getitem__(
        self,
        inputs: Union[Any, List[Any]]
    ):
        ...