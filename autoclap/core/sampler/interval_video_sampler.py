import cv2

from autoclap.core.sampler import BaseVideoSampler

class IntervalVideoSampler(BaseVideoSampler):
    """Sampling every specific frame."""
    def __init__(
        self,
        video: str,
        batch_size: int,
        interval: int,
        drop_last: bool = True,
    ):
        super().__init__(video, batch_size, drop_last)
        self.interval = interval
    
    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx % self.interval == 0:
                yield idx, frame

            idx += 1

        cap.release()