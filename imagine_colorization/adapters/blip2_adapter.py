"""BLIP-2 captioning adapter."""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from imagine_colorization.config import Blip2Config


class Blip2Captioner:
    """Wraps BLIP-2 model loading and caption generation."""

    def __init__(self, config: Blip2Config) -> None:
        self.config = config
        self._loaded = False
        self._model = None
        self._processor = None
        self._torch = None
        self._image_cls = None

    def _resolve_dtype(self, torch_module):
        name = self.config.dtype.lower()
        if name == "float16":
            return torch_module.float16
        if name == "bfloat16":
            return torch_module.bfloat16
        return torch_module.float32

    def _to_pil(self, image: np.ndarray):
        if self._image_cls is None:
            from PIL import Image

            self._image_cls = Image
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return self._image_cls.fromarray(image.astype("uint8"), mode="RGB")

    def _sanitize_caption(self, caption: str) -> str:
        cleaned = caption.strip()
        if not cleaned:
            return cleaned
        patterns = [
            r"\\bblack and white\\b",
            r"\\bblack-and-white\\b",
            r"\\bmonochrome\\b",
            r"\\bgrayscale\\b",
            r"\\bdesaturated\\b",
            r"\\bcolorless\\b",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s+", " ", cleaned)
        cleaned = re.sub(r"\\s+([,.;:])", r"\\1", cleaned)
        return cleaned.strip()

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except Exception as exc:  # pragma: no cover - depends on external libs
            raise RuntimeError("无法导入 BLIP-2 依赖，请先安装 transformers/torch/pillow。") from exc

        dtype = self._resolve_dtype(torch)
        local_only = self.config.local_files_only or os.path.isdir(self.config.model_name)
        self._processor = Blip2Processor.from_pretrained(
            self.config.model_name,
            local_files_only=local_only,
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        self._model.to(self.config.device)
        self._model.eval()
        self._torch = torch
        self._loaded = True

    def generate(self, image: np.ndarray) -> str:
        self.load()
        pil_image = self._to_pil(image)
        prompts = []
        if self.config.prompt:
            prompts.append(self.config.prompt)
        prompts.extend(
            [
                "Question: Describe the image without mentioning colors. Answer:",
                "Describe the image.",
            ]
        )
        for prompt in prompts:
            caption = self._generate_with_prompt(pil_image, prompt)
            caption = self._sanitize_caption(caption)
            if caption:
                return caption
        return ""

    def _generate_with_prompt(self, image, prompt: str) -> str:
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens
            )
        captions = self._processor.batch_decode(output, skip_special_tokens=True)
        return captions[0]

    def unload(self) -> None:
        """Release model weights to free GPU memory."""

        if not self._loaded:
            return
        self._model = None
        self._processor = None
        if self._torch is not None and self.config.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
        self._loaded = False
