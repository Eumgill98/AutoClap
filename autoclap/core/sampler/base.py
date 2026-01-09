from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple, List

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