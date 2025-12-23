import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imagine_colorization.colorizer import ColorizationModel
from imagine_colorization.config import ColorizationConfig, ControlNetConfig
from imagine_colorization.types import ColorizationSample, ReferenceComposition


def _load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取灰度图: {path}")
    return image


def _load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取参考图: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_caption(text: Optional[str], path: Optional[str]) -> str:
    if text:
        return text.strip()
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return ""


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="ControlNet-based colorization demo.")
    parser.add_argument("--gray", required=True, help="灰度输入图路径")
    parser.add_argument("--reference", required=True, help="Refined reference 图路径")
    parser.add_argument("--out", default="outputs/colorized.png", help="输出图路径")
    parser.add_argument(
        "--mode",
        default="sd15_controlnet",
        choices=["sd15_controlnet", "hint_propagation"],
        help="Colorization backend",
    )
    parser.add_argument("--caption", default=None, help="正向提示词（覆盖 caption 文件）")
    parser.add_argument("--caption-file", default=None, help="caption.txt 路径")
    parser.add_argument("--prompt-template", default="{caption}, colorized")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-source", choices=["grayscale", "reference"], default="grayscale")
    parser.add_argument("--controlnet-weights", default=None)
    parser.add_argument("--controlnet-config", default="ControlNet/models/cldm_v15.yaml")
    parser.add_argument("--controlnet-repo", default="ControlNet")
    parser.add_argument("--controlnet-device", default="cuda")
    parser.add_argument("--controlnet-control-type", default="canny")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--guess-mode", action="store_true")
    parser.add_argument("--save-memory", action="store_true")
    parser.add_argument("--prompt-prefix", default="best quality, extremely detailed, realistic")
    parser.add_argument("--hint-coarse-size", type=int, default=64)
    parser.add_argument("--hint-refine-strength", type=float, default=0.7)
    parser.add_argument("--hint-refine-blur", type=float, default=9.0)
    parser.add_argument("--propagation-radius", type=int, default=8)
    parser.add_argument("--propagation-eps", type=float, default=1e-3)
    parser.add_argument("--propagation-sigma-color", type=float, default=12.0)
    parser.add_argument("--propagation-sigma-space", type=float, default=12.0)
    parser.add_argument("--propagation-strength", type=float, default=1.0)
    parser.add_argument("--save-hint-map", default=None, help="保存 hint 颜色图")
    args = parser.parse_args(argv)

    gray = _load_grayscale(args.gray)
    reference = _load_rgb(args.reference)
    caption = _load_caption(args.caption, args.caption_file)

    if args.mode == "sd15_controlnet" and not args.controlnet_weights:
        raise ValueError("mode=sd15_controlnet 需要 --controlnet-weights")

    controlnet_cfg = ControlNetConfig(
        repo_path=args.controlnet_repo,
        sd_config_path=args.controlnet_config,
        controlnet_weights_path=args.controlnet_weights or "",
        device=args.controlnet_device,
        control_type=args.controlnet_control_type,
        image_resolution=args.image_resolution,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        guess_mode=args.guess_mode,
        negative_prompt=args.negative_prompt or ControlNetConfig.negative_prompt,
        prompt_prefix=args.prompt_prefix,
        save_memory=args.save_memory,
    )
    color_cfg = ColorizationConfig(
        mode=args.mode,
        controlnet=controlnet_cfg,
        color_prompt_template=args.prompt_template,
        control_source=args.control_source,
        hint_coarse_size=args.hint_coarse_size,
        hint_refine_strength=args.hint_refine_strength,
        hint_refine_blur=args.hint_refine_blur,
        propagation_radius=args.propagation_radius,
        propagation_eps=args.propagation_eps,
        propagation_sigma_color=args.propagation_sigma_color,
        propagation_sigma_space=args.propagation_sigma_space,
        propagation_strength=args.propagation_strength,
    )

    sample = ColorizationSample(grayscale=gray, caption=caption)
    if args.seed is not None:
        sample.metadata["seed"] = args.seed

    composition = ReferenceComposition(
        image=reference,
        mask=np.ones(reference.shape[:2], dtype="float32")[..., None],
        provenance={"source": 0},
    )

    model = ColorizationModel(color_cfg)
    outputs = model(sample, composition)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(outputs.colorized, cv2.COLOR_RGB2BGR))
    if args.save_hint_map and outputs.hint_map is not None:
        cv2.imwrite(args.save_hint_map, cv2.cvtColor(outputs.hint_map, cv2.COLOR_RGB2BGR))
    print(f"colorized image saved to {args.out}")


if __name__ == "__main__":
    main()
