from typing import Union, List, Tuple, Callable, Optional
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

    def crop_by_bbox(
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

    def _one_run(
        self,
        frame_bboxes: FrameBboxes,
        add_pre: Optional[Callable] = None,
        **kwargs,
    ) -> List[OCROutput]: 
        """
        Run one frame inference.

        Args:
            frame_bboxes (FrameBboxes): One frame and bboxes that are of clapperboard.
            add_pre (Optional[Callable]) = Additional Preprocessing methods. Default is None.
        
        Returns:
            List of OCROutput.
        """

        frame, bbox = frame_bboxes
        frame = self.crop_by_bbox(frame, bbox)
        
        if add_pre is not None:
            frame = add_pre(frame)
        
        outputs = self.model(frame, **kwargs)

        return outputs

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
        results = []
        if not isinstance(frames_bboxes, list):
            frames_bboxes = [frames_bboxes]
        
        if verbose:
            print("="*80)
            print(f"TOTAL RUN OCR FRAMES: {len(frames_bboxes)}")

        for frame_bboxes in frames_bboxes:
            results.extend(self._one_run(
                frame_bboxes,
                add_pre,
                **kwargs,
            ))
        
        if verbose:
            print("="*80)
            
        return results