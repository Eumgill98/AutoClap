from typing import List, Union
from tqdm import tqdm
from pathlib import Path

from autoclap.detector.base import BaseDetector
from autoclap.core.output import DetectorPipelineOutput
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
    ) -> DetectorPipelineOutput:
        """
        Run one video inference.

        Args:
            video_sampler (BaseVideoSampler): Video sampler for sampling strategy
            verbose (bool): show progress. Default is True.
            
        Returns:
            DetectorPipelineOutput structured detection information.
        """
        results = []
        frame_indices = []
 
        iterator = video_sampler
        if verbose:
            file_name = Path(video_sampler.video).stem
            iterator = tqdm(video_sampler, desc=f"Running {file_name}_video_detection", unit="batch")

        for batch in iterator:
            indices = [idx for idx, _ in batch]
            frames = [frame for _, frame in batch]

            # inference
            outs = self.model(frames, **kwargs)
            
            results.extend(outs)
            frame_indices.extend(indices)
        
        return DetectorPipelineOutput(
            video_path=video_sampler.video,
            video_sampler=video_sampler.name,
            frame_indices=frame_indices,
            detections=results,
            fps=video_sampler.fps,
            total_frames=video_sampler.total_frames,
        )

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
    ) -> List[DetectorPipelineOutput]:
        """
        Run videos inference.

        Args:
            video_samplers (Union[BaseVideoSampler, List[BaseVideoSampler]]): Video samplers for sampling strategy
            verbose (bool): show progress. Default is True.
            
        Returns:
            List of List that DetectorOutput containing structured detection information.
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
        