from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

Frame = Any
SampledFrame = Tuple[int, Frame]

class BaseVideoSampler(ABC):
    """A abstract class for video sampler"""

    @abstractmethod
    def sample(self, video: Any) -> Iterable[SampledFrame]:
        """
        Args:
            video: video source (path, cv2.VideoCaputre, stream, etc)
        
        Yields:
            (frame_index, frame)
        """
        ...