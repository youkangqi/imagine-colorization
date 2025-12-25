import argparse
import json
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imagine_colorization.config import (
    Blip2Config,
    ColorizationConfig,
    ControlNetConfig,
    DinoV2Config,
    ImaginationConfig,
    PipelineConfig,
    RefinementConfig,
    SemanticSamConfig,
)
from imagine_colorization.pipeline import ImagineColorizationPipeline
from imagine_colorization.types import ColorizationSample

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, artifacts, oil painting, painterly, overly saturated, cartoon, illustration"
)
AVOID_GRAYSCALE = "monochrome, grayscale, black and white, desaturated"
DEFAULT_PROMPT_PREFIX = (
    "realistic, photographic, natural lighting, translucent colors, soft saturation, true-to-life, high detail"
)


def _merge_negative_prompt(base: Optional[str]) -> str:
    if not base:
        return AVOID_GRAYSCALE
    tokens = [token.strip() for token in base.split(",") if token.strip()]
    existing = {token.lower() for token in tokens}
    for token in AVOID_GRAYSCALE.split(","):
        cleaned = token.strip()
        if cleaned and cleaned.lower() not in existing:
            tokens.append(cleaned)
    return ", ".join(tokens)


def _augment_blip2_prompt(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return prompt
    lower = prompt.lower()
    if "avoid mentioning" in lower or "monochrome" in lower or "grayscale" in lower:
        return prompt
    if "Answer:" in prompt:
        head, tail = prompt.split("Answer:", 1)
        head = head.rstrip()
        tail = tail.strip()
        return f"{head} Avoid mentioning: {AVOID_GRAYSCALE}. Answer: {tail}".strip()
    return f"{prompt.strip()} Avoid mentioning: {AVOID_GRAYSCALE}."

def _load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取灰度图: {path}")
    return image


def _save_image(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.ndim == 2:
        cv2.imwrite(path, image.astype("uint8"))
        return
    if image.shape[-1] == 3:
        cv2.imwrite(path, cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2BGR))
        return
    cv2.imwrite(path, image.astype("uint8"))


def _save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _save_masks(masks: List[np.ndarray], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for idx, mask in enumerate(masks):
        path = os.path.join(outdir, f"segment_{idx:03d}.png")
        _save_image(path, (mask.astype("uint8") * 255))


def _write_metadata(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end Imagine-Colorization demo.")
    parser.add_argument("--gray", required=True, help="灰度输入图路径")
    parser.add_argument("--outdir", default="outputs/pipeline", help="输出目录")
    parser.add_argument("--caption", default=None, help="手动指定 caption")

    parser.add_argument("--prompt-template", default="{caption}")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--append-negative-to-prompt", action="store_true")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1, help="每次采样的候选数")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--save-memory", action="store_true")

    parser.add_argument("--controlnet-weights", required=True)
    parser.add_argument("--controlnet-config", default="ControlNet/models/cldm_v15.yaml")
    parser.add_argument("--controlnet-repo", default="ControlNet")
    parser.add_argument("--controlnet-device", default="cuda")
    parser.add_argument("--controlnet-control-type", default="canny")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--guess-mode", action="store_true")

    parser.add_argument("--blip2-model", default=None)
    parser.add_argument("--blip2-device", default="cuda")
    parser.add_argument("--blip2-dtype", default="float16")
    parser.add_argument(
        "--blip2-prompt",
        default="Question: Describe the image without mentioning colors. Answer:",
    )

    parser.add_argument("--semantic-sam-repo", default="Semantic-SAM")
    parser.add_argument(
        "--semantic-sam-config",
        default="Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml",
    )
    parser.add_argument("--semantic-sam-ckpt", default="model_Seg/swint_only_sam_many2many.pth")
    parser.add_argument("--semantic-sam-model-type", default="T")
    parser.add_argument("--semantic-sam-device", default="cuda")
    parser.add_argument("--semantic-sam-resize", type=int, default=640)
    parser.add_argument("--max-segments", type=int, default=5)
    parser.add_argument("--min-segment-area", type=int, default=256)
    parser.add_argument("--overlap-thresh", type=float, default=0.9)
    parser.add_argument("--mask-blur", type=float, default=5.0)

    parser.add_argument("--dino-ckpt", default="model_Seg/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dino-repo", default="dinov2")
    parser.add_argument("--dino-arch", default="dinov2_vitl14")
    parser.add_argument("--dino-input-size", type=int, default=518)
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--dino-dtype", default="float16")

    parser.add_argument(
        "--color-mode",
        default="hint_propagation",
        choices=["hint_propagation", "sd15_controlnet"],
    )
    parser.add_argument("--color-prompt-template", default="{caption}, colorized")
    parser.add_argument("--color-prompt-prefix", default=None)
    parser.add_argument("--color-negative-prompt", default=None)
    parser.add_argument("--color-control-source", default="grayscale", choices=["grayscale", "reference"])
    parser.add_argument("--color-controlnet-weights", default=None)
    parser.add_argument("--color-controlnet-config", default=None)
    parser.add_argument("--color-controlnet-repo", default=None)
    parser.add_argument("--color-controlnet-device", default=None)
    parser.add_argument("--color-controlnet-control-type", default=None)
    parser.add_argument("--color-image-resolution", type=int, default=None)
    parser.add_argument("--color-steps", type=int, default=None)
    parser.add_argument("--color-guidance-scale", type=float, default=None)
    parser.add_argument("--color-strength", type=float, default=None)
    parser.add_argument("--color-guess-mode", action="store_true")
    parser.add_argument("--color-save-memory", action="store_true")

    parser.add_argument("--hint-coarse-size", type=int, default=64)
    parser.add_argument("--hint-refine-strength", type=float, default=0.7)
    parser.add_argument("--hint-refine-blur", type=float, default=9.0)
    parser.add_argument("--hint-max-per-segment", type=int, default=10)
    parser.add_argument("--propagation-radius", type=int, default=8)
    parser.add_argument("--propagation-eps", type=float, default=1e-3)
    parser.add_argument("--propagation-sigma-color", type=float, default=12.0)
    parser.add_argument("--propagation-sigma-space", type=float, default=12.0)
    parser.add_argument("--propagation-strength", type=float, default=1.0)
    parser.add_argument("--save-json", action="store_true", help="保存 metadata.json")

    args = parser.parse_args(argv)

    args.negative_prompt = _merge_negative_prompt(args.negative_prompt)
    if args.color_negative_prompt is not None:
        args.color_negative_prompt = _merge_negative_prompt(args.color_negative_prompt)
    args.blip2_prompt = _augment_blip2_prompt(args.blip2_prompt)

    if args.color_mode == "sd15_controlnet" and not (
        args.color_controlnet_weights or args.controlnet_weights
    ):
        raise ValueError("color_mode=sd15_controlnet 需要 controlnet 权重")

    image_name = os.path.splitext(os.path.basename(args.gray))[0]
    outdir = os.path.join(args.outdir, image_name)
    candidates_dir = os.path.join(outdir, "candidates")
    segments_dir = os.path.join(outdir, "segments")

    imagination_controlnet = ControlNetConfig(
        repo_path=args.controlnet_repo,
        sd_config_path=args.controlnet_config,
        controlnet_weights_path=args.controlnet_weights,
        device=args.controlnet_device,
        control_type=args.controlnet_control_type,
        image_resolution=args.image_resolution,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        guess_mode=args.guess_mode,
        negative_prompt=args.negative_prompt,
        prompt_prefix=args.prompt_prefix,
        save_memory=args.save_memory,
    )
    # print(f"BLIP2 prompt:{args.blip2_prompt}")
    imagination_cfg = ImaginationConfig(
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        prompt_template=args.prompt_template,
        negative_prompt=args.negative_prompt,
        append_negative_prompt=args.append_negative_to_prompt,
        seed=args.seed,
        controlnet=imagination_controlnet,
        blip2=Blip2Config(
            model_name=args.blip2_model,
            device=args.blip2_device,
            dtype=args.blip2_dtype,
            prompt=args.blip2_prompt,
            local_files_only=True,
        )
        if args.blip2_model
        else None,
    )

    semantic_cfg = SemanticSamConfig(
        repo_path=args.semantic_sam_repo,
        config_path=args.semantic_sam_config,
        checkpoint_path=args.semantic_sam_ckpt,
        model_type=args.semantic_sam_model_type,
        device=args.semantic_sam_device,
        resize_short_edge=args.semantic_sam_resize,
    )
    dino_cfg = DinoV2Config(
        checkpoint_path=args.dino_ckpt,
        repo_path=args.dino_repo,
        arch=args.dino_arch,
        input_size=args.dino_input_size,
        device=args.dino_device,
        dtype=args.dino_dtype,
    )
    refinement_cfg = RefinementConfig(
        semantic_sam=semantic_cfg,
        dino=dino_cfg,
        max_segments=args.max_segments,
        segment_min_area=args.min_segment_area,
        overlap_thresh=args.overlap_thresh,
        mask_blur=args.mask_blur,
    )

    color_controlnet_weights = args.color_controlnet_weights or args.controlnet_weights
    color_controlnet = ControlNetConfig(
        repo_path=args.color_controlnet_repo or args.controlnet_repo,
        sd_config_path=args.color_controlnet_config or args.controlnet_config,
        controlnet_weights_path=color_controlnet_weights,
        device=args.color_controlnet_device or args.controlnet_device,
        control_type=args.color_controlnet_control_type or args.controlnet_control_type,
        image_resolution=args.color_image_resolution or args.image_resolution,
        steps=args.color_steps or args.steps,
        guidance_scale=args.color_guidance_scale or args.guidance_scale,
        strength=args.color_strength or args.strength,
        guess_mode=args.color_guess_mode or args.guess_mode,
        negative_prompt=args.color_negative_prompt or args.negative_prompt,
        prompt_prefix=args.color_prompt_prefix or args.prompt_prefix,
        save_memory=args.color_save_memory or args.save_memory,
    )

    color_cfg = ColorizationConfig(
        mode=args.color_mode,
        controlnet=color_controlnet,
        color_prompt_template=args.color_prompt_template,
        control_source=args.color_control_source,
        hint_coarse_size=args.hint_coarse_size,
        hint_refine_strength=args.hint_refine_strength,
        hint_refine_blur=args.hint_refine_blur,
        hint_max_per_segment=args.hint_max_per_segment,
        propagation_radius=args.propagation_radius,
        propagation_eps=args.propagation_eps,
        propagation_sigma_color=args.propagation_sigma_color,
        propagation_sigma_space=args.propagation_sigma_space,
        propagation_strength=args.propagation_strength,
    )

    pipeline_cfg = PipelineConfig(
        imagination=imagination_cfg,
        refinement=refinement_cfg,
        colorization=color_cfg,
    )

    sample = ColorizationSample(grayscale=_load_grayscale(args.gray), caption=args.caption)
    if args.color_negative_prompt or args.negative_prompt:
        sample.metadata["negative_prompt"] = args.color_negative_prompt or args.negative_prompt
    if args.seed is not None:
        sample.metadata["seed"] = args.seed

    pipeline = ImagineColorizationPipeline(pipeline_cfg)
    outputs = pipeline(sample)

    os.makedirs(outdir, exist_ok=True)
    caption_path = os.path.join(outdir, "caption.txt")
    source_path = os.path.join(outdir, "source.png")
    control_image_path = os.path.join(outdir, "control_image.png")
    refined_path = os.path.join(outdir, "refined.png")
    refined_mask_path = os.path.join(outdir, "refined_mask.png")
    colorized_path = os.path.join(outdir, "colorized.png")
    hint_map_path = os.path.join(outdir, "hint_map.png")
    metadata_path = os.path.join(outdir, "metadata.json")

    _save_image(source_path, sample.grayscale)
    _save_text(caption_path, outputs.imagination.caption)

    candidate_paths: List[str] = []
    for idx, candidate in enumerate(outputs.imagination.candidates):
        path = os.path.join(candidates_dir, f"candidate_{idx:03d}.png")
        _save_image(path, candidate.image)
        candidate_paths.append(path)
    if outputs.imagination.candidates and outputs.imagination.candidates[0].control_image is not None:
        _save_image(control_image_path, outputs.imagination.candidates[0].control_image)

    _save_image(refined_path, outputs.refinement.composition.image)
    _save_image(refined_mask_path, outputs.refinement.composition.mask[..., 0] * 255)

    masks = pipeline.refinement.compute_masks(sample)
    _save_masks(masks, segments_dir)

    _save_image(colorized_path, outputs.colorization.colorized)
    if outputs.colorization.hint_map is not None:
        _save_image(hint_map_path, outputs.colorization.hint_map)

    if args.save_json:
        candidate_metadata = []
        for candidate, path in zip(outputs.imagination.candidates, candidate_paths):
            candidate_metadata.append(
                {
                    "path": path,
                    "score": candidate.score,
                    "seed": candidate.seed,
                }
            )
        metadata = {
            "args": vars(args),
            "caption": outputs.imagination.caption,
            "provenance": outputs.refinement.composition.provenance,
            "selected_indices": outputs.refinement.selected_indices,
            "candidates": candidate_metadata,
            "outputs": {
                "caption": caption_path,
                "source": source_path,
                "control_image": control_image_path,
                "candidates": candidate_paths,
                "refined": refined_path,
                "refined_mask": refined_mask_path,
                "segments": segments_dir,
                "colorized": colorized_path,
                "hint_map": hint_map_path if outputs.colorization.hint_map is not None else None,
            },
        }
        _write_metadata(metadata_path, metadata)

    print(f"Done. Outputs saved to {outdir}")


if __name__ == "__main__":
    main()
