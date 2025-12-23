"""End-to-end Imagine-Colorization pipeline skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from imagine_colorization.colorizer import ColorizationModel, ColorizationOutputs
from imagine_colorization.config import PipelineConfig
from imagine_colorization.imagination import ImaginationModule, ImaginationOutputs
from imagine_colorization.refinement import ReferenceRefinementModule, RefinementOutputs
from imagine_colorization.types import ColorizationSample


@dataclass
class PipelineOutputs:
    """Aggregate outputs from each stage."""

    imagination: ImaginationOutputs
    refinement: RefinementOutputs
    colorization: ColorizationOutputs


class ImagineColorizationPipeline:
    """Orchestrates the imagination, refinement, and colorization stages.

    Flow (following the paper):
        1) Imagination: caption the grayscale input and sample diverse colorful
           references from a pre-trained diffusion model (or alternative
           generator).
        2) Reference Refinement: align references, select the best segments
           using semantic + structural cues, and compose a high-quality
           guidance image.
        3) Colorization: condition a reference-guided colorization network on
           the composed reference to obtain the final result. The explicit
           reference enables iterative, localized edits—users can swap segments
           in the composition and re-run stage 3 without resampling.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.imagination = ImaginationModule(self.config.imagination)
        self.refinement = ReferenceRefinementModule(self.config.refinement)
        self.colorizer = ColorizationModel(self.config.colorization)

    def __call__(self, sample: ColorizationSample) -> PipelineOutputs:
        imagination_outputs = self.imagination(sample)
        if sample.caption is None:
            sample.caption = imagination_outputs.caption
        refinement_outputs = self.refinement(imagination_outputs.candidates, sample)
        colorization_outputs = self.colorizer(sample, refinement_outputs.composition)

        return PipelineOutputs(
            imagination=imagination_outputs,
            refinement=refinement_outputs,
            colorization=colorization_outputs,
        )
