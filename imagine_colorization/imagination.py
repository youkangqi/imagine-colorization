"""Imagination module skeleton.

The imagination module mirrors the paper's strategy: understand the grayscale
scene, turn that understanding into colorful hypotheses via a strong generative
prior, and keep track of candidates for downstream refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from imagine_colorization.config import ImaginationConfig
from imagine_colorization.types import ColorizationSample, ReferenceCandidate


@dataclass
class ImaginationOutputs:
    """Outputs from the imagination stage."""

    caption: str
    candidates: List[ReferenceCandidate]


class ImaginationModule:
    """Generates colorful reference candidates guided by text prompts."""

    def __init__(self, config: ImaginationConfig) -> None:
        self.config = config

    def describe_scene(self, sample: ColorizationSample) -> str:
        """Derive a text caption for the grayscale image.

        This placeholder mirrors the paper's use of an automatic captioner to
        expose semantic priors to the diffusion model. Replace with BLIP/CLIP
        captioner for production use.
        """

        if sample.caption:
            return sample.caption
        return "A detailed photograph matching the grayscale input."

    def generate_candidates(self, caption: str, sample: ColorizationSample) -> Iterable[np.ndarray]:
        """Yield raw colorful images from a generative model.

        Substitute the body with a Stable Diffusion sampler conditioned on the
        caption and structural priors (depth/edges) extracted from the grayscale
        input.
        """

        for _ in range(self.config.num_candidates):
            # Placeholder: return a tiled grayscale image as a colorful dummy.
            yield np.repeat(sample.grayscale[..., None], 3, axis=-1)

    def score_candidate(self, candidate: np.ndarray, caption: str, sample: ColorizationSample) -> float:
        """Compute a semantic-structural score for ranking candidates."""

        # Placeholder: downstream refinement will handle structure; here we use
        # a dummy uniform score.
        return 1.0

    def __call__(self, sample: ColorizationSample) -> ImaginationOutputs:
        caption = self.describe_scene(sample)
        candidates: List[ReferenceCandidate] = []
        for image in self.generate_candidates(caption, sample):
            score = self.score_candidate(image, caption, sample)
            candidates.append(ReferenceCandidate(image=image, caption=caption, score=score))
        return ImaginationOutputs(caption=caption, candidates=candidates)
