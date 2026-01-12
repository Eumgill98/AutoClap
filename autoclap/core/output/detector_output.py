from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field

class DetectorOutput(BaseModel):
    """Detector model structured output (a one video)."""
    video_path: str = Field(..., description="video file path.")
    video_sampler: str = Field(..., description="use video sampler.")

    frame_indices: List[int] = Field(default_factory=list, description="Frame index.")
    detections: List[Dict[str, Any]] = Field(default_factory=list, description="Detector model result.")

    fps: int = Field(..., description="Original video fps.")
    total_frames: float = Field(..., description="Original video total frames.")

    def to_json(self, **kwargs) -> str:
        """Return JSON string."""
        return self.model_dump_json(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation."""
        return self.model_dump()
    
    def get_frame_best_score(
        self, 
        frame_det: Dict[str, Any], 
        conf_th: float = 0.5
    ) -> Tuple[int, Optional[List[int]]]:
        """
        Convert detection results one frame.

        Args:
            frame_det: {
                "boxes": [[x1,y1,x2,y2], ...],
                "scores": [float, ...]
                ...
            }
            conf_th (float): confidence score threshold.

        Returns:
            (score, box | None)
        """

        if not frame_det["scores"]:
            return 0.0, None
        
        best_idx = None
        best_score = 0.0

        for i, score in enumerate(frame_det["scores"]):
            if score < conf_th:
                continue
            if score > best_score:
                best_score = float(score)
                best_idx = i
        
        if best_idx is None:
            return 0.0, None
        
        return (best_score, frame_det["boxes"][best_idx])

    def get_all_frame_best_score(
        self,
        conf_th: float = 0.5,
        ) -> List[Tuple[int, Tuple[int, Optional[List[int]]]]]:
        """
        Convert detection results into a per-frame score representation.

        Args:
            conf_th (float): confidence score threshold.

        Returns:
            List of tuples:
                [(frame_index, (best_class_score, list_of_bboxes)), ...]
        """
        result = []

        if not self.detections:
            return result
        
        for idx, det in zip(self.indices, self.detections):
            best = self.get_best_score_in_frame(det, conf_th=conf_th)
            result.append((idx, best))

    def get_clapperboard_zone(
        self,
        conf_th: float = 0.5,
        min_duration: int = 1,
    ) -> List[Tuple[float, float, bool, Optional[List[int]]]]:
        """
        Divide the video into time zones where clapperboard is present or absent.

        Args:
            conf_th (float): Minimum confidence to consider clapperboard present.
            min_duration (int): Minimum consecutive frames to consider a valid zone.

        Returns:
            List of tuples: [(start_time, end_time, presence_flag, bbox), ...]
                start_time, end_time: in seconds
                presence_flag: True if clapperboard is present, False otherwise
                bbox: bbox of the first frame of the zone if present, else None
        """
        zones = []

        zone_start_idx = 0
        presence = None
        start_bbox = None

        scores = self.get_all_frame_best_score(conf_th=conf_th)

        for i, (frame_idx, (score, bbox)) in enumerate(scores):
            current_presence = score >= conf_th

            if presence is None:
                presence = current_presence
                zone_start_idx = frame_idx
                start_bbox = bbox if presence else None
            elif current_presence != presence:
                if frame_idx - zone_start_idx >= min_duration:
                    start_time = zone_start_idx / self.fps
                    end_time = (frame_idx - 1) / self.fps
                    zones.append((start_time, end_time, presence, start_bbox))
                zone_start_idx = frame_idx
                presence = current_presence
                start_bbox = bbox if presence else None

        if zone_start_idx <= scores[-1][0]:
            start_time = zone_start_idx / self.fps
            end_time = scores[-1][0] / self.fps
            zones.append((start_time, end_time, presence, start_bbox))

        return zones