import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imagine_colorization.config import Blip2Config, ControlNetConfig, ImaginationConfig
from imagine_colorization.imagination import ImaginationModule
from imagine_colorization.types import ColorizationSample


def _load_grayscale(path: str):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def _ensure_path_exists(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    return path


def _build_config(args: argparse.Namespace) -> ImaginationConfig:
    controlnet_kwargs = {
        "control_type": args.control_type,
        "sd_config_path": args.sd_config,
        "controlnet_weights_path": args.controlnet_weights,
        "device": args.controlnet_device,
        "image_resolution": args.image_resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "guess_mode": args.guess_mode,
        "save_memory": args.save_memory,
    }
    if args.negative_prompt is not None:
        controlnet_kwargs["negative_prompt"] = args.negative_prompt
    controlnet = ControlNetConfig(**controlnet_kwargs)

    imagination_kwargs = {
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "prompt_template": args.prompt_template,
        "controlnet": controlnet,
    }
    if args.negative_prompt is not None:
        imagination_kwargs["negative_prompt"] = args.negative_prompt
    if args.blip2_model is not None:
        imagination_kwargs["blip2"] = Blip2Config(
            model_name=args.blip2_model,
            device=args.blip2_device,
            dtype=args.blip2_dtype,
            max_new_tokens=args.blip2_max_new_tokens,
            prompt=args.blip2_prompt,
            local_files_only=args.blip2_local_files_only,
        )
    return ImaginationConfig(**imagination_kwargs)


def _save_images(images, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for idx, image in enumerate(images):
        path = os.path.join(outdir, f"candidate_{idx:03d}.png")
        cv2.imwrite(path, image)

def _save_caption(caption: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "caption.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(caption.strip() + "\n")


def _save_control_image(image: np.ndarray, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "control_image.png")
    cv2.imwrite(path, image)


def _generate_in_batches(
    module: ImaginationModule,
    sample: ColorizationSample,
    total: int,
    batch_size: int,
    seed: Optional[int],
) -> Tuple[str, list]:
    candidates = []
    caption = sample.caption or ""
    offset = 0
    while len(candidates) < total:
        remaining = total - len(candidates)
        current = min(batch_size, remaining)
        module.config.num_candidates = current
        if seed is not None:
            module.config.seed = seed + offset
        outputs = module(sample)
        caption = outputs.caption
        candidates.extend(outputs.candidates)
        offset += current
    return caption, candidates

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="生成多张上色候选图（SD1.5 + ControlNet）")
    parser.add_argument("--image", required=True, help="黑白输入图片路径")
    parser.add_argument("--outdir", default="outputs/candidates", help="输出目录")
    parser.add_argument("--prompt", default=None, help="可选文本提示")
    parser.add_argument("--num-candidates", type=int, default=4, help="候选数量")
    parser.add_argument("--batch-size", type=int, default=1, help="每次生成的张数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--control-type", default="canny", choices=["canny", "hed", "depth", "seg"])
    parser.add_argument("--sd-config", default="ControlNet/models/cldm_v15.yaml")
    parser.add_argument("--controlnet-weights", default="ControlNet/models/control_sd15_canny.pth")
    parser.add_argument("--controlnet-device", default="cuda", help="ControlNet 设备")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--guess-mode", action="store_true")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--prompt-template", default="{caption}")
    parser.add_argument("--save-memory", action="store_true", help="启用低显存模式")
    parser.add_argument("--blip2-model", default=None, help="BLIP-2 模型名称")
    parser.add_argument("--blip2-device", default="cuda", help="BLIP-2 设备")
    parser.add_argument("--blip2-dtype", default="float16", help="BLIP-2 dtype")
    parser.add_argument("--blip2-max-new-tokens", type=int, default=30, help="BLIP-2 生成长度")
    parser.add_argument("--blip2-prompt", default=None, help="BLIP-2 提示词")
    parser.add_argument("--blip2-local-files-only", action="store_true", help="仅从本地加载 BLIP-2")
    args = parser.parse_args(argv)

    _ensure_path_exists(args.sd_config)
    _ensure_path_exists(args.controlnet_weights)

    grayscale = _load_grayscale(args.image)
    config = _build_config(args)
    module = ImaginationModule(config)
    sample = ColorizationSample(grayscale=grayscale, caption=args.prompt)
    if sample.caption is None:
        sample.caption = module.describe_scene(sample)

    caption, candidates = _generate_in_batches(
        module,
        sample,
        total=args.num_candidates,
        batch_size=max(1, args.batch_size),
        seed=args.seed,
    )
    images = [candidate.image for candidate in candidates]
    _save_images(images, args.outdir)
    if candidates and candidates[0].control_image is not None:
        _save_control_image(candidates[0].control_image, args.outdir)
    _save_caption(caption, args.outdir)


if __name__ == "__main__":
    main()
