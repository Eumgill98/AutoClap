from typing import Union, List, Tuple
import numpy as np 
import cv2

from autoclap.ocr.base import BaseOCR

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
        pad: int = 0,
        pad_value: int | tuple = 0,
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

        if pad > 0:
            if crop.ndim == 2:
                pad_val = pad_value if isinstance(pad_value, int) else pad_value[0]
            else:
                pad_val = pad_value if isinstance(pad_value, tuple) else (pad_value,) * 3

            crop = cv2.copyMakeBorder(
                crop,
                pad, pad, pad, pad,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_val,
            )

        return crop

    def crop_by_bbox_batch(
        self,
        frames: List[np.ndarray],
        bboxes: List[List[int | float]],
        pad: int = 0,
        pad_value: int | tuple = 0,
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
            crop = self._crop_by_bbox(frame, bbox, pad, pad_value)
            crops.append(crop)

        return crops

    def run(
        self,
        frames_bboxes: Union[FrameBboxes, List[FrameBboxes]],
        verbose: bool = True,
        pad: int = 0,
        pad_value: int | tuple = 0,
        **kwargs, 
    ):
        """
        Run OCRpipeline inference.

        Args:
            frames_bboxes (Union[np.array, List[np.array]]): List of frames and bboxes
            verbose (bool): show progress.
            pad (int): padding size. 
            pad_value (int or tuple): padding fill value.

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
        
        frames = self.crop_by_bbox_batch(frames, bboxes, pad, pad_value)
        results = self.model(frames, **kwargs) 

        if verbose:
            print("="*80)
            
        return results