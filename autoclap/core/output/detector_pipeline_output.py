from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import cv2

from autoclap.core.output import DetectorOutput

class DetectorPipelineOutput(BaseModel):
    """Detector pipeline structured output (a one video)."""
    video_path: str = Field(..., description="video file path.")
    video_sampler: str = Field(..., description="use video sampler.")

    frame_indices: List[int] = Field(default_factory=list, description="Frame index.")
    detections: List[DetectorOutput] = Field(default_factory=list, description="Detector model result.")

    fps: int = Field(..., description="Original video fps.")
    total_frames: float = Field(..., description="Original video total frames.")

    def to_json(self, **kwargs) -> str:
        """Return JSON string."""
        return self.model_dump_json(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation."""
        return self.model_dump()
    
    def get_frame_by_time(self, time: float) -> np.ndarray:
        """
        Get a specific frame from the video by time.

        Args:
            time (float): frame time

        Returns:
            np.ndarray: BGR image (H, W, C)

        Raises:
            ValueError: if frame_idx is negative
            RuntimeError: if frame cannot be read
        """
        frame_idx = self.time2frame(time, self.fps)

        return (self.get_frame(frame_idx=frame_idx))
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame from the video.

        Args:
            frame_idx (int): 0-based frame index

        Returns:
            np.ndarray: BGR image (H, W, C)

        Raises:
            ValueError: if frame_idx is negative
            RuntimeError: if frame cannot be read
        """
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

        return frame
    
    def time2frame(self, time: float, fps:float) -> int:
        """
        Convert time(seconds) to frame index.

        Args:
        time (float): Time in seconds.
        fps (float): Frames per second.

        Returns:
            int: Frame index (0-based).
        """
        if time < 0:
            raise ValueError("time must be non-negative")
        if fps <= 0:
            raise ValueError("fps must be positive")

        return int(round(time * fps))
    
    def get_frame_best_score(
        self, 
        frame_det: DetectorOutput, 
        conf_th: float = 0.5
    ) -> Tuple[int, Optional[List[int]]]:
        """
        Convert detection results one frame.

        Args:
            frame_det: Detection.
            conf_th (float): confidence score threshold.

        Returns:
            (score, box | None)
        """

        if not frame_det:
            return 0.0, []
        
        best_idx = None
        best_score = 0.0

        for i, det in enumerate(frame_det.detections):
            if det.score < conf_th:
                continue
            if det.score > best_score:
                best_score = float(det.score)
                best_idx = i
        
        if best_idx is None:
            return 0.0, []
        
        return (best_score, frame_det.detections[best_idx].bbox)

    def get_all_frame_best_score(
        self,
        conf_th: float = 0.5,
        ) -> List[Tuple[int, Tuple[int, List[int]]]]:
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
        
        for idx, det in zip(self.frame_indices, self.detections):
            best = self.get_frame_best_score(det, conf_th=conf_th)
            result.append((idx, best))
            
        return result

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