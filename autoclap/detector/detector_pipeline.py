from typing import List, Dict, Any, Union
from tqdm import tqdm
from pathlib import Path

from autoclap.detector.base import BaseDetector
from autoclap.core.sampler import BaseVideoSampler

class DetectorPipeline:
    """A class for detector pipeline."""
    def __init__(
        self,
        model: BaseDetector,
    ):
        self.model: BaseDetector = model

    def _one_run(
        self,
        video_sampler: BaseVideoSampler,
        verbose: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run one video inference.

        Args:
            video_sampler (BaseVideoSampler): Video sampler for sampling strategy
            verbose (bool): show progress. Default is True.
            
        Returns:
            List of Dict containing structured detection information.
        """
        results = []

        iterator = video_sampler
        if verbose:
            file_name = Path(video_sampler.video).stem
            iterator = tqdm(video_sampler, desc=f"Running {file_name}_video_detection", unit="batch")

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
        video_samplers: Union[BaseVideoSampler, List[BaseVideoSampler]],
        verbose: bool = True,
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run videos inference.

        Args:
            video_samplers (Union[BaseVideoSampler, List[BaseVideoSampler]]): Video samplers for sampling strategy
            verbose (bool): show progress. Default is True.
            
        Returns:
            List of List that is dicts containing structured detection information.
        """
        results = []

        if not isinstance(video_samplers, list):
            video_samplers = [video_samplers]

        if verbose:
            print("="*80)
            print(f"TOTAL RUN VIDEO: {len(video_samplers)}")

        for video_sampler in video_samplers:
            results.append(self._one_run(video_sampler=video_sampler, verbose=verbose, **kwargs))

        if verbose:
            print("="*80)

        return results
        