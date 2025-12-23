"""DINOv2 feature extractor adapter."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from imagine_colorization.config import DinoV2Config


class DinoV2Adapter:
    """Extracts robust universal features using DINOv2."""

    def __init__(self, config: DinoV2Config) -> None:
        self.config = config
        self._loaded = False
        self._model = None
        self._processor = None
        self._preprocess = None
        self._torch = None

    def _resolve_dtype(self, torch_module):
        name = self.config.dtype.lower()
        if name == "float16":
            return torch_module.float16
        if name == "bfloat16":
            return torch_module.bfloat16
        return torch_module.float32

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        return os.path.abspath(path)

    def _load_with_torch_hub(self) -> None:
        import torch

        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path:
            raise RuntimeError("Missing DINOv2 checkpoint path.")
        checkpoint_path = self._resolve_path(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint_path}")

        repo_path = self.config.repo_path
        if repo_path and os.path.isdir(repo_path):
            model = torch.hub.load(repo_path, self.config.arch, source="local", pretrained=False)
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.config.arch, pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.to(self.config.device)
        model.eval()

        try:
            from torchvision import transforms
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Missing torchvision/PIL for DINOv2 preprocessing.") from exc

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        preprocess = transforms.Compose(
            [
                transforms.Resize(self.config.input_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self._model = model
        self._preprocess = preprocess
        self._torch = torch
        self._loaded = True

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, Dinov2Model
        except Exception as exc:  # pragma: no cover - depends on external libs
            torch = None
            AutoImageProcessor = None
            Dinov2Model = None

        if self.config.checkpoint_path:
            self._load_with_torch_hub()
            return

        if torch is None or AutoImageProcessor is None or Dinov2Model is None:
            raise RuntimeError("Failed to import DINOv2 dependencies.")

        local_only = self.config.local_files_only or os.path.isdir(self.config.model_name)
        dtype = self._resolve_dtype(torch)
        self._processor = AutoImageProcessor.from_pretrained(
            self.config.model_name,
            local_files_only=local_only,
        )
        self._model = Dinov2Model.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        self._model.to(self.config.device)
        self._model.eval()
        self._torch = torch
        self._loaded = True

    def extract(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract a single embedding for a masked region."""

        self.load()
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if mask is not None:
            mask = mask.astype(bool)
            if mask.sum() == 0:
                raise ValueError("Empty mask provided to DINOv2 extractor.")
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            cropped = image[y0:y1, x0:x1].copy()
            mask_crop = mask[y0:y1, x0:x1]
            cropped[~mask_crop] = 0
            image = cropped

        if self._processor is not None:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]
        else:
            tensor = self._preprocess(image).unsqueeze(0).to(self.config.device)
            with self._torch.no_grad():
                outputs = self._model(tensor)
            embedding = outputs[:, 0]
        return embedding.squeeze(0).detach().cpu().numpy()
