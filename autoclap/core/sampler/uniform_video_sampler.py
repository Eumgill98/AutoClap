import cv2
import math

from autoclap.core.sampler import BaseVideoSampler

class UniformVideoSampler(BaseVideoSampler):
    """Uniformly sample frames across the video"""

    def __init__(
        self,
        video: str,
        batch_size: int,
        num_samples: int,
        drop_last: bool = True,
    ):
        super().__init__(video, batch_size, drop_last)
        self.num_samples = num_samples

    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return

        effective_samples = min(self.num_samples, total_frames)
        step = total_frames / effective_samples
        target_indices = {int(i * step) for i in range(effective_samples)}

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx in target_indices:
                yield idx, frame

            idx += 1

        cap.release()

    def __len__(self) -> int:
        total_frames = self.total_frames

        if total_frames <= 0:
            raise TypeError("Cannot determine video length")

        effective_samples = min(self.num_samples, total_frames)

        if self.drop_last:
            return effective_samples // self.batch_size
        return math.ceil(effective_samples / self.batch_size)
