import cv2
import math

from autoclap.core.sampler import BaseVideoSampler

class SequentialVideoSampler(BaseVideoSampler):
    """Load every frame."""

    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)
        idx = 0

        while cap.isOpened():
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
        
        if self.drop_last:
            return total_frames // self.batch_size
        return math.ceil(total_frames / self.batch_size)