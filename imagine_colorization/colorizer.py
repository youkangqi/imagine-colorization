"""Reference-guided colorization module skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from imagine_colorization.adapters.controlnet_adapter import ControlNetAdapter
from imagine_colorization.config import ColorizationConfig
from imagine_colorization.types import ColorizationSample, ReferenceComposition
from imagine_colorization.utils.vision import normalize_uint8, resize, to_uint8


@dataclass
class ColorizationOutputs:
    """Outputs from the colorization stage."""

    colorized: np.ndarray
    low_level_features: np.ndarray
    reference_features: np.ndarray
    hint_map: Optional[np.ndarray] = None


class ColorizationModel:
    """Decode grayscale inputs into color using a refined reference."""

    def __init__(self, config: ColorizationConfig) -> None:
        self.config = config
        self._controlnet: Optional[ControlNetAdapter] = None
        if self.config.mode == "sd15_controlnet":
            self._controlnet = ControlNetAdapter(self.config.controlnet)

    def encode_grayscale(self, grayscale: np.ndarray) -> np.ndarray:
        """Encode grayscale input to multi-scale features."""

        gray = normalize_uint8(grayscale)
        # Placeholder: single-scale features.
        return gray[..., None]

    def encode_reference(self, reference: ReferenceComposition) -> np.ndarray:
        """Encode the refined reference for cross-attention or fusion."""

        ref = normalize_uint8(reference.image)
        return ref

    def _build_prompt(self, sample: ColorizationSample) -> str:
        template = self.config.color_prompt_template
        caption = sample.caption or ""
        prompt = template.format(caption=caption, palette="")
        parts = [part.strip() for part in prompt.split(",") if part.strip()]
        return ", ".join(parts)

    def _build_control_image(
        self, sample: ColorizationSample, composition: ReferenceComposition
    ) -> np.ndarray:
        if self.config.control_source == "reference":
            return composition.image
        return sample.grayscale

    def _generate_with_controlnet(
        self, sample: ColorizationSample, composition: ReferenceComposition
    ) -> np.ndarray:
        if self._controlnet is None:
            raise RuntimeError("ControlNet adapter is not initialized.")
        prompt = self._build_prompt(sample)
        control_source = self._build_control_image(sample, composition)
        control_image = self._controlnet.build_control_image(control_source)
        negative_prompt = sample.metadata.get("negative_prompt")
        seed = sample.metadata.get("seed")
        images = self._controlnet.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_samples=1,
            seed=seed,
        )
        return images[0]

    def _resize_with_cv2(
        self, image: np.ndarray, size: Tuple[int, int], interpolation: int
    ) -> np.ndarray:
        try:
            import cv2
        except Exception:
            return resize(image, size)
        return cv2.resize(image, (size[1], size[0]), interpolation=interpolation)

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except Exception:
            return mask.astype("float32")
        radius = max(1, int(round(self.config.hint_refine_blur)))
        kernel = radius * 2 + 1
        blurred = cv2.GaussianBlur(mask.astype("float32"), (kernel, kernel), 0)
        if blurred.max() > 0:
            blurred = blurred / blurred.max()
        return np.clip(blurred, 0.0, 1.0)

    def _generate_hint_lab(
        self, sample: ColorizationSample, composition: ReferenceComposition
    ) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Hint color generation requires OpenCV.") from exc

        gray = sample.grayscale
        reference = composition.image
        base_size = gray.shape[:2]
        if reference.shape[:2] != base_size:
            reference = self._resize_with_cv2(reference, base_size, cv2.INTER_LINEAR)

        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype("float32")
        short_edge = min(base_size)
        target_short = min(self.config.hint_coarse_size, short_edge)
        scale = target_short / float(short_edge)
        coarse_size = (max(1, int(round(base_size[0] * scale))), max(1, int(round(base_size[1] * scale))))
        coarse_lab = self._resize_with_cv2(ref_lab, coarse_size, cv2.INTER_LINEAR)
        coarse_lab = self._resize_with_cv2(coarse_lab, base_size, cv2.INTER_LINEAR)

        hint_lab = coarse_lab.copy()
        mask = composition.mask
        if mask is not None:
            mask_map = mask[..., 0] if mask.ndim == 3 else mask
            if mask_map.shape[:2] != base_size:
                mask_map = self._resize_with_cv2(mask_map, base_size, cv2.INTER_LINEAR)
            mask_map = np.clip(mask_map, 0.0, 1.0)
            if self.config.hint_refine_blur > 0:
                mask_map = self._smooth_mask(mask_map)
            blend = np.clip(self.config.hint_refine_strength, 0.0, 1.0)
            weight = (mask_map * blend)[..., None]
            hint_lab[..., 1:] = (
                (1.0 - weight) * coarse_lab[..., 1:] + weight * ref_lab[..., 1:]
            )
        return hint_lab

    def _propagate_hints(
        self, sample: ColorizationSample, hint_lab: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Hint propagation requires OpenCV.") from exc

        gray = sample.grayscale
        base_size = gray.shape[:2]
        if hint_lab.shape[:2] != base_size:
            hint_lab = self._resize_with_cv2(hint_lab, base_size, cv2.INTER_LINEAR)
        hint_lab = hint_lab.astype("float32")
        hint_ab = hint_lab[..., 1:]

        guide = gray.astype("float32") / 255.0
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            radius = int(self.config.propagation_radius)
            eps = float(self.config.propagation_eps)
            a = cv2.ximgproc.guidedFilter(guide, hint_ab[..., 0] / 255.0, radius, eps)
            b = cv2.ximgproc.guidedFilter(guide, hint_ab[..., 1] / 255.0, radius, eps)
            prop_ab = np.stack([a, b], axis=-1) * 255.0
        else:
            sigma_color = float(self.config.propagation_sigma_color)
            sigma_space = float(self.config.propagation_sigma_space)
            a = cv2.bilateralFilter(hint_ab[..., 0], 0, sigma_color, sigma_space)
            b = cv2.bilateralFilter(hint_ab[..., 1], 0, sigma_color, sigma_space)
            prop_ab = np.stack([a, b], axis=-1)

        prop_ab = np.clip(prop_ab, 0.0, 255.0)
        strength = np.clip(self.config.propagation_strength, 0.0, 1.0)
        merged_ab = hint_ab + strength * (prop_ab - hint_ab)
        lab = np.dstack([gray.astype("float32"), merged_ab]).astype("float32")
        rgb = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2RGB)
        hint_rgb = cv2.cvtColor(hint_lab.astype("uint8"), cv2.COLOR_LAB2RGB)
        return rgb, hint_rgb

    def fuse(self, gray_features: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
        """Fuse grayscale and reference features."""

        if self.config.use_cross_attention:
            # Placeholder: concatenate along channel axis.
            return np.concatenate([gray_features, reference_features], axis=-1)
        return gray_features

    def decode(self, fused_features: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """Decode fused features back to an RGB image."""

        # Placeholder: tile the luminance and blend simple chroma hints.
        height, width = output_shape
        luminance = fused_features[..., 0]
        rgb = np.repeat(luminance[..., None], 3, axis=-1)
        return to_uint8(rgb)

    def __call__(self, sample: ColorizationSample, composition: ReferenceComposition) -> ColorizationOutputs:
        gray_features = self.encode_grayscale(sample.grayscale)
        reference_features = self.encode_reference(composition)
        hint_map = None
        if self.config.mode == "sd15_controlnet":
            colorized = self._generate_with_controlnet(sample, composition)
        elif self.config.mode == "hint_propagation":
            hint_lab = self._generate_hint_lab(sample, composition)
            colorized, hint_map = self._propagate_hints(sample, hint_lab)
        else:
            fused = self.fuse(gray_features, reference_features)
            colorized = self.decode(fused, output_shape=sample.grayscale.shape[:2])
        return ColorizationOutputs(
            colorized=colorized,
            low_level_features=gray_features,
            reference_features=reference_features,
            hint_map=hint_map,
        )
