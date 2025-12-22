# Imagine-Colorization

This repository accompanies the paper [Automatic Controllable Colorization via Imagination](https://xy-cong.github.io/imagine-colorization/).

## Method recap

The paper introduces a three-stage pipeline:

1. **Imagination Module** — Caption the grayscale input and sample diverse colorful references from a strong pre-trained generator (e.g., diffusion), exposing semantic priors that resemble human imagination.
2. **Reference Refinement Module** — Align candidates to the grayscale structure, rank them with semantic/structural cues, and compose a high-quality, instance-aware reference by selecting the best regions.
3. **Reference-Guided Colorization** — Decode color with a reference-conditioned network. Because references are explicit, users can swap or edit regions and re-run colorization without regenerating everything.

## Code skeleton

The `imagine_colorization` package captures the engineering scaffolding for the above method without binding to a specific deep learning framework. Each class exposes clean interfaces so concrete implementations can drop in later:

```
imagine_colorization/
├── __init__.py               # Public API
├── colorizer.py              # Reference-guided colorization network wrapper
├── config.py                 # Dataclass configurations for all modules
├── imagination.py            # Captioning + diffusion/GAN sampling
├── pipeline.py               # End-to-end orchestrator
├── refinement.py             # Reference refinement and composition
└── utils/
    └── vision.py             # Lightweight vision utilities
```

Example usage:

```python
import numpy as np
from imagine_colorization import ImagineColorizationPipeline, ColorizationSample

pipeline = ImagineColorizationPipeline()
sample = ColorizationSample(grayscale=np.zeros((256, 256), dtype=np.uint8))
outputs = pipeline(sample)

colorized = outputs.colorization.colorized  # Final RGB image (uint8)
reference = outputs.refinement.composition.image  # Composed colorful reference
```

Replace the placeholders in each module with production-ready components (e.g., BLIP captioner, Stable Diffusion sampler, optical flow aligner, and a reference-guided UNet/Transformer) to build a full system.
