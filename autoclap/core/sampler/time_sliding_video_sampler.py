import cv2
import math

from autoclap.core.sampler import BaseVideoSampler


class TimeSlidingVideoSampler(BaseVideoSampler):
    """
    Slide over video by time (seconds / minutes) and sample key frames.
    """

    def __init__(
        self,
        video: str,
        batch_size: int,
        window_sec: float = 2.0,
        stride_sec: float = 1.0,
        key_pos: str = "center", 
        drop_last: bool = True,
    ):
        super().__init__(video, batch_size, drop_last)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.key_pos = key_pos

    def _iter_frames(self):
        cap = cv2.VideoCapture(self.video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return

        window_frames = int(self.window_sec * fps)
        stride_frames = int(self.stride_sec * fps)

        if window_frames <= 0 or stride_frames <= 0:
            raise ValueError("window_sec and stride_sec must be > 0")

        start = 0
        while start + window_frames <= total_frames:
            if self.key_pos == "start":
                frame_idx = start
            elif self.key_pos == "end":
                frame_idx = start + window_frames - 1
            else:  # center
                frame_idx = start + window_frames // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            yield frame_idx, frame
            start += stride_frames

        cap.release()

    def __len__(self) -> int:
        fps, total_frames = self.fps, self.total_frames

        if fps <= 0 or total_frames <= 0:
            raise TypeError("Cannot determine video length")

        window_frames = int(self.window_sec * fps)
        stride_frames = int(self.stride_sec * fps)

        if window_frames <= 0 or stride_frames <= 0:
            raise ValueError("window_sec and stride_sec must be > 0")

        num_samples = max(0, (total_frames - window_frames) // stride_frames + 1)

        if self.drop_last:
            return num_samples // self.batch_size
        return math.ceil(num_samples / self.batch_size)