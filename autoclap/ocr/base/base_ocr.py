from abc import ABC, abstractmethod
from typing import List, Any, Union, Optional
from pydantic import BaseModel, Field, ConfigDict

import torch

from autoclap.core.output import OCROutput

class BaseOCR(BaseModel, ABC):
    """A abstract class for OCR model"""
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    ln: Union[str, List[str]] = Field(default="en", description="Language(s) used for OCR")
    ocr: Optional[Any] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def predict(
        self,
        images: List[Any],
        **kwargs,
    ):
        ...

    @abstractmethod
    def structure_output(
        self,
        outputs: List[Any]
    ) -> List[OCROutput]:
        ...

    def __call__(
        self,
        inputs: Union[Any, List[Any]],
        **kwargs,
    ) -> List[OCROutput]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        outputs = self.predict(inputs, **kwargs)
        return self.structure_output(outputs)