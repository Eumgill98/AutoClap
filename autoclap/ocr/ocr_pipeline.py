from typing import Union, List, Tuple, Callable, Optional, Sequence
import numpy as np 

from autoclap.ocr.base import BaseOCR
from autoclap.core.output import OCROutput

FrameBboxes = Tuple[np.array, List[int]]

class OCRPipeline:
    """A class for ocr pipeline."""
    def __init__(
        self,
        model: BaseOCR,
    ):
        self.model: BaseOCR = model

    def _crop_by_bbox(
        self,
        frame: np.ndarray,
        bbox: list[int | float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox}")

        crop = frame[y1:y2, x1:x2].copy()

        if crop.shape[0] < 16 or crop.shape[1] < 16:
            raise ValueError(f"Crop too small: {crop.shape}")

        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)

        return crop
    
    def _batch_preprocessing(
        self,
        frames: List[np.array],
        add_pre: Callable,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Multi image add prprocessing
        """
        result: List[np.ndarray] = []

        for frame in frames:
            new_frame = add_pre(frame, **kwargs)
            result.append(new_frame)

        return result

    def crop_by_bbox_batch(
        self,
        frames: List[np.ndarray],
        bboxes: List[List[int | float]],
    ) -> List[np.ndarray]:
        """
        Multi image crop processing
        """
        if len(frames) != len(bboxes):
            raise ValueError(
                f"frames ({len(frames)}) and bboxes ({len(bboxes)}) length mismatch"
            )

        crops: List[np.ndarray] = []

        for frame, bbox in zip(frames, bboxes):
            crop = self._crop_by_bbox(frame, bbox)
            crops.append(crop)

        return crops

    def run(
        self,
        frames_bboxes: Union[FrameBboxes, List[FrameBboxes]],
        add_pre: Optional[Callable] = None,
        verbose: bool = True,
        **kwargs, 
    ):
        """
        Run OCRpipeline inference.

        Args:
            frames_bboxes (Union[np.array, List[np.array]]): List of frames and bboxes
            add_pre (Optional[Callable]): Additional Preprocessing methods. Default is None.
            verbose (bool): show progress.

        Returns:
            List of OCROutput.
        """
        if not isinstance(frames_bboxes, list):
            frames_bboxes = [frames_bboxes]
        
        frames = [frame for frame, _ in frames_bboxes]
        bboxes = [bbox for _, bbox in frames_bboxes]

        if verbose:
            print("="*80)
            print(f"TOTAL RUN OCR FRAMES: {len(frames_bboxes)}")
        
        frames = self.crop_by_bbox_batch(frames, bboxes)

        if add_pre is not None:
            frames = self._batch_preprocessing(
                frames=frames,
                add_pre=add_pre,
                **kwargs,
            )
    
        results = self.model(frames, **kwargs) 

        if verbose:
            print("="*80)
            
        return results