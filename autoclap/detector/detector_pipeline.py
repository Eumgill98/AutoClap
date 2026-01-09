from typing import List, Dict, Any
from tqdm import tqdm

from autoclap.detector.base import BaseDetector
from autoclap.core.sampler import BaseVideoSampler

class DetectorPipeline:
    """A class for detector pipeline."""
    def __init__(
        self,
        model: BaseDetector,
    ):
        self.model: BaseDetector = model

    def to(
        self,
        device: str,
    ):
        """
        Move the model to the specified device

        Args: 
            device (str): Target device ('cpu', 'cuda' ...)
        """
        self.model.to(device=device)

    def run(
        self,
        video_sampler: BaseVideoSampler,
        verbose: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run video inference.

        Args:
            video_sampler (BaseVideoSampler): Video sampler for sampling strategy
            verbose (bool): show progress. Default is True.
            
        Returns:
            List of Dict containing structured detection information.
        """
        results = []

        iterator = video_sampler
        if verbose:
            iterator = tqdm(video_sampler, desc="Running detection", unit="batch")

        for batch in iterator:
            frame_indices = [idx for idx, _ in batch]
            frames = [frame for _, frame in batch]

            # inference
            outs = self.model(frames, **kwargs)

            for idx, det in zip(frame_indices, outs):
                results.append({
                    "frame_index": idx,
                    "detections": det,
                })

        return results