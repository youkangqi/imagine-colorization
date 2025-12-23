import argparse
import glob
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imagine_colorization.config import DinoV2Config, RefinementConfig, SemanticSamConfig
from imagine_colorization.refinement import ReferenceRefinementModule
from imagine_colorization.types import ColorizationSample, ReferenceCandidate


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


def _collect_candidates(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    filtered: List[str] = []
    for path in paths:
        name = os.path.basename(path)
        if name.startswith("control_image"):
            continue
        if name.startswith("refined"):
            continue
        filtered.append(path)
    paths = filtered
    if not paths:
        raise FileNotFoundError(f"未找到候选参考图: {pattern}")
    return paths


def _save_segmentation_masks(masks: List[np.ndarray], outdir: str) -> None:
    if not masks:
        return
    os.makedirs(outdir, exist_ok=True)
    for idx, mask in enumerate(masks):
        path = os.path.join(outdir, f"segment_{idx:03d}.png")
        cv2.imwrite(path, (mask.astype("uint8") * 255))


def _build_config(args: argparse.Namespace) -> RefinementConfig:
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
    return RefinementConfig(
        semantic_sam=semantic_cfg,
        dino=dino_cfg,
        max_segments=args.max_segments,
        segment_min_area=args.min_segment_area,
        distance_metric=args.distance_metric,
        mask_blur=args.mask_blur,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="最小化参考细化脚本（Semantic-SAM + DINOv2）")
    parser.add_argument("--gray", required=True, help="灰度输入图路径")
    parser.add_argument("--candidates", required=True, help="候选参考图 glob，例如 outputs/candidates/*.png")
    parser.add_argument("--out", default="outputs/refined.png", help="输出参考图路径")
    parser.add_argument("--semantic-sam-repo", default="Semantic-SAM", help="Semantic-SAM 子模块路径")
    parser.add_argument(
        "--semantic-sam-config",
        default="Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml",
        help="Semantic-SAM 配置文件路径",
    )
    parser.add_argument(
        "--semantic-sam-ckpt",
        default="model_Seg/swint_only_sam_many2many.pth",
        help="Semantic-SAM 权重路径",
    )
    parser.add_argument("--semantic-sam-model-type", default="T", help="Semantic-SAM 模型类型 L/T")
    parser.add_argument("--semantic-sam-device", default="cuda", help="Semantic-SAM 设备")
    parser.add_argument("--semantic-sam-resize", type=int, default=640, help="Semantic-SAM 短边缩放")
    parser.add_argument("--max-segments", type=int, default=5, help="最大使用 segment 数")
    parser.add_argument("--min-segment-area", type=int, default=256, help="segment 最小像素面积")
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "l1", "l2"])
    parser.add_argument("--mask-blur", type=float, default=5.0, help="分割融合的 mask 模糊半径")
    parser.add_argument("--dino-ckpt", default="model_Seg/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dino-repo", default="dinov2")
    parser.add_argument("--dino-arch", default="dinov2_vitl14")
    parser.add_argument("--dino-input-size", type=int, default=518)
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--dino-dtype", default="float16")
    parser.add_argument("--segments-outdir", default=None, help="保存分割结果目录")
    args = parser.parse_args(argv)

    gray = _load_grayscale(args.gray)
    candidate_paths = _collect_candidates(args.candidates)
    candidates = [
        ReferenceCandidate(image=_load_rgb(path), caption="", score=None) for path in candidate_paths
    ]
    sample = ColorizationSample(grayscale=gray)

    config = _build_config(args)
    module = ReferenceRefinementModule(config)
    outputs = module(candidates, sample)
    refined = outputs.composition.image
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(refined, cv2.COLOR_RGB2BGR))
    if args.segments_outdir:
        masks = module.compute_masks(sample)
        _save_segmentation_masks(masks, args.segments_outdir)
    print(f"refined reference saved to {args.out}")


if __name__ == "__main__":
    main()
