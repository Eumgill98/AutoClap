from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class Detection(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    score: float = Field(..., description="Confience Score")
    class_id: int = Field(..., description="Class index")
    class_name: str = Field(..., description="Class name")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

class DetectorOutput(BaseModel):
    "Detector model structured output (a one frame)."
    detections: List[Detection] = Field(default_factory=list, description="Detection info")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)