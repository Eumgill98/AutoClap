import cv2
import math

from autoclap.core.sampler import BaseVideoSampler

class HeadVideoSampler(BaseVideoSampler):
    """Sampling up to N frames from the front"""

    def __init__(
        self,
        video: str,
        batch_size: int,
        max_frames: int,
        drop_last: bool = True,
    ):
        super().__init__(video, batch_size, drop_last)
        self.max_frames = max_frames

    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)
        idx = 0

        while cap.isOpened() and idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            yield idx, frame
            idx += 1

        cap.release()
    
    def __len__(self) -> int:
        cap = cv2.VideoCapture(self.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0:
            raise TypeError("Cannot determine video length")

        effective_frames = min(total_frames, self.max_frames)

        if self.drop_last:
            return effective_frames // self.batch_size
        return math.ceil(effective_frames / self.batch_size)