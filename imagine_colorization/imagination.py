"""Imagination module skeleton.

The imagination module mirrors the paper's strategy: understand the grayscale
scene, turn that understanding into colorful hypotheses via a strong generative
prior, and keep track of candidates for downstream refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from imagine_colorization.adapters.blip2_adapter import Blip2Captioner
from imagine_colorization.adapters.controlnet_adapter import ControlNetAdapter
from imagine_colorization.config import ImaginationConfig
from imagine_colorization.types import ColorizationSample, ReferenceCandidate


@dataclass
class ImaginationOutputs:
    """Outputs from the imagination stage."""

    caption: str
    candidates: List[ReferenceCandidate]


class ImaginationModule:
    """Generates colorful reference candidates guided by text prompts."""

    _NEGATIVE_APPEND_EXCLUDES = {
        "monochrome",
        "grayscale",
        "black and white",
        "desaturated",
    }

    def __init__(self, config: ImaginationConfig) -> None:
        self.config = config
        self.controlnet = ControlNetAdapter(config.controlnet)
        self.captioner = Blip2Captioner(config.blip2) if config.blip2 else None

    def _build_prompt(self, caption: str) -> str:
        try:
            return self.config.prompt_template.format(caption=caption)
        except KeyError as exc:
            raise ValueError("prompt_template must include '{caption}'.") from exc

    def _resolve_negative_prompt(self) -> str:
        if self.config.negative_prompt:
            return self.config.negative_prompt
        return self.config.controlnet.negative_prompt

    def _negative_for_prompt(self, negative_prompt: str) -> str:
        tokens = [token.strip() for token in negative_prompt.split(",")]
        filtered = [
            token
            for token in tokens
            if token and token.lower() not in self._NEGATIVE_APPEND_EXCLUDES
        ]
        return ", ".join(filtered)

    def _prepare_control_image(self, sample: ColorizationSample) -> np.ndarray:
        if sample.edges is not None:
            return self.controlnet.prepare_control_image(sample.edges)
        return self.controlnet.build_control_image(sample.grayscale)

    def describe_scene(self, sample: ColorizationSample) -> str:
        """Derive a text caption for the grayscale image.

        This placeholder mirrors the paper's use of an automatic captioner to
        expose semantic priors to the diffusion model. Replace with BLIP/CLIP
        captioner for production use.
        """

        if sample.caption:
            return sample.caption
        if self.captioner is not None:
            caption = self.captioner.generate(sample.grayscale)
            self.captioner.unload()
            if caption and caption.strip():
                return caption
        return "A detailed photograph matching the grayscale input."

    def generate_candidates(
        self, caption: str, sample: ColorizationSample
    ) -> Iterable[ReferenceCandidate]:
        """Generate reference candidates using SD1.5 + ControlNet."""

        negative_prompt = self._resolve_negative_prompt()
        prompt = self._build_prompt(caption)
        if self.config.append_negative_prompt and negative_prompt:
            filtered = self._negative_for_prompt(negative_prompt)
            if filtered:
                prompt = f"{prompt}. Without: {filtered}"
        control_image = self._prepare_control_image(sample)
        remaining = max(1, self.config.num_candidates)
        batch_size = max(1, self.config.batch_size)
        batch_idx = 0
        while remaining > 0:
            current = min(batch_size, remaining)
            batch_seed = None
            if self.config.seed is not None:
                batch_seed = self.config.seed + batch_idx
            images = self.controlnet.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                num_samples=current,
                seed=batch_seed,
            )
            for image in images:
                yield ReferenceCandidate(
                    image=image,
                    caption=caption,
                    control_image=control_image,
                    seed=batch_seed,
                )
            remaining -= current
            batch_idx += 1

    def score_candidate(self, candidate: ReferenceCandidate, sample: ColorizationSample) -> float:
        """Compute a semantic-structural score for ranking candidates."""

        # Placeholder: downstream refinement will handle structure; here we use
        # a dummy uniform score.
        _ = sample
        return 1.0

    def __call__(self, sample: ColorizationSample) -> ImaginationOutputs:
        caption = self.describe_scene(sample)
        candidates: List[ReferenceCandidate] = list(self.generate_candidates(caption, sample))
        for candidate in candidates:
            candidate.score = self.score_candidate(candidate, sample)
        return ImaginationOutputs(caption=caption, candidates=candidates)
