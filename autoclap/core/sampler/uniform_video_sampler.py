import cv2

from autoclap.core.sampler import BaseVideoSampler

class UniformVideoSampler(BaseVideoSampler):
    """Sample every N frames"""

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
        if total_frames == 0:
            cap.release()
            return
        
        step = total_frames / self.num_samples
        target_indices = {int(i * step) for i in range(self.num_samples)}

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx in target_indices:
                yield idx, frame

            idx += 1

        cap.release()
