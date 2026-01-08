import cv2

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