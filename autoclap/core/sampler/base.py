from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple, List
import cv2

Frame = Any
SampledFrame = Tuple[int, Frame]
Batch = List[SampledFrame]

class BaseVideoSampler(ABC):
    """Iterable video sampler (batched)"""

    def __init__(
        self,
        video: str,
        batch_size: int,
        drop_last: bool = True,
    ):
        """
        Args:
            video (str): video file path
            batch_size (int): num of loading frames
            drop_last (bool): To drop remaining frames. Default is True
        """
        self.video = video
        self.batch_size = batch_size
        self.drop_last = drop_last

        # metadata
        self._fps = 0
        self._total_frames = 0

        self._set_meta()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def total_frames(self) -> float:
        return self._total_frames
    
    @property
    def fps(self) -> int:
        return self._fps

    def _set_meta(self) -> float:
        """Set metadata"""
        cap = cv2.VideoCapture(self.video)
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @abstractmethod
    def _iter_frames(self) -> Iterable[SampledFrame]:
        """
        Yields:
            (frame_index, frame)
        """
        ...

    def __iter__(self) -> Iterable[Batch]:
        batch: Batch = []

        for sample in self._iter_frames():
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch