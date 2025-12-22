"""ControlNet adapter wrapping the submodule implementation."""

from __future__ import annotations

import os
import random
import sys
from typing import List, Optional

import numpy as np

from imagine_colorization.config import ControlNetConfig


class ControlNetAdapter:
    """Thin wrapper around the ControlNet submodule for inference."""

    def __init__(self, config: ControlNetConfig) -> None:
        self.config = config
        self._loaded = False
        self._model = None
        self._sampler = None
        self._detector = None
        self._torch = None
        self._einops = None
        self._resize_image = None
        self._hwc3 = None

    def _ensure_repo_on_path(self) -> None:
        repo_path = os.path.abspath(self.config.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _build_detector(self) -> None:
        if self.config.control_type == "canny":
            from annotator.canny import CannyDetector

            self._detector = CannyDetector()
            return
        raise NotImplementedError(f"Unsupported control_type: {self.config.control_type}")

    def load(self) -> None:
        """Load ControlNet model and annotator dependencies."""

        if self._loaded:
            return
        self._ensure_repo_on_path()
        try:
            from annotator.util import HWC3, resize_image
            from cldm.ddim_hacked import DDIMSampler
            from cldm.model import create_model, load_state_dict
            import einops
            import torch
        except Exception as exc:  # pragma: no cover - depends on external libs
            raise RuntimeError("Failed to import ControlNet dependencies.") from exc

        model = create_model(self.config.sd_config_path).cpu()
        model.load_state_dict(
            load_state_dict(self.config.controlnet_weights_path, location=self.config.device)
        )
        model = model.to(self.config.device)

        self._model = model
        self._sampler = DDIMSampler(model)
        self._torch = torch
        self._einops = einops
        self._resize_image = resize_image
        self._hwc3 = HWC3
        self._build_detector()
        self._loaded = True

    def build_control_image(self, image: np.ndarray) -> np.ndarray:
        """Build a ControlNet condition image from a grayscale or RGB input."""

        self.load()
        img = self._hwc3(image)
        img = self._resize_image(img, self.config.image_resolution)
        if self.config.control_type == "canny":
            detected = self._detector(
                img, self.config.canny_low_threshold, self.config.canny_high_threshold
            )
            return self._hwc3(detected)
        raise NotImplementedError(f"Unsupported control_type: {self.config.control_type}")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        control_image: np.ndarray,
        num_samples: int,
        seed: Optional[int],
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        guess_mode: Optional[bool] = None,
    ) -> List[np.ndarray]:
        """Generate samples with SD1.5 + ControlNet."""

        self.load()
        torch = self._torch
        einops = self._einops

        if seed is None:
            seed = random.randint(0, 65535)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        steps = steps or self.config.steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        strength = strength or self.config.strength
        guess_mode = self.config.guess_mode if guess_mode is None else guess_mode
        negative_prompt = negative_prompt or self.config.negative_prompt
        prompt_prefix = self.config.prompt_prefix
        if prompt_prefix:
            prompt = f"{prompt}, {prompt_prefix}"

        control_image = self._hwc3(control_image)
        control = torch.from_numpy(control_image.copy()).float().to(self.config.device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        height, width = control_image.shape[:2]
        shape = (4, height // 8, width // 8)

        model = self._model
        if self.config.save_memory and hasattr(model, "low_vram_shift"):
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)],
        }

        if self.config.save_memory and hasattr(model, "low_vram_shift"):
            model.low_vram_shift(is_diffusing=True)

        if hasattr(model, "control_scales"):
            if guess_mode:
                model.control_scales = [
                    strength * (0.825 ** float(12 - i)) for i in range(13)
                ]
            else:
                model.control_scales = [strength] * 13

        samples, _ = self._sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=self.config.eta,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
        )

        if self.config.save_memory and hasattr(model, "low_vram_shift"):
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5
        ).cpu().numpy()
        x_samples = np.clip(x_samples, 0, 255).astype(np.uint8)

        return [x_samples[i] for i in range(num_samples)]
