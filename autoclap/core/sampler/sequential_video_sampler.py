import cv2

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