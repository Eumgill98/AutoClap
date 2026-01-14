from pydantic import BaseModel, Field, ConfigDict
from typing import List

class OCRText(BaseModel):
    text: str = Field(..., description="text")
    confidence: float = Field(..., description="confidence score")

    model_config = ConfigDict(arbitrary_types_allowed=True)



class OCROutput(BaseModel):
    texts :List[OCRText] = Field(default_factory=list, description="A frame detected textes")

    model_config = ConfigDict(arbitrary_types_allowed=True)