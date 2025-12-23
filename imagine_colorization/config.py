"""Configuration dataclasses for the Imagine-Colorization pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ControlNetConfig:
    """Settings for SD1.5 + ControlNet inference.

    Attributes:
        repo_path: Path to the ControlNet submodule.
        sd_config_path: Path to the SD1.5 config file.
        controlnet_weights_path: Path to the ControlNet weights.
        device: Torch device string.
        precision: "fp16" or "fp32".
        control_type: Which ControlNet annotator to use (canny/hed/depth/seg).
        image_resolution: Resize resolution for the control image.
        steps: Sampler steps.
        guidance_scale: Classifier-free guidance scale.
        strength: ControlNet strength.
        guess_mode: Whether to enable guess mode.
        eta: DDIM eta.
        negative_prompt: Default negative prompt.
        prompt_prefix: Default prompt prefix for quality.
        canny_low_threshold: Canny low threshold.
        canny_high_threshold: Canny high threshold.
        save_memory: Whether to enable low VRAM mode if supported.
    """

    repo_path: str = "ControlNet"
    sd_config_path: str = "ControlNet/models/cldm_v15.yaml"
    controlnet_weights_path: str = "ControlNet/models/control_sd15_canny.pth"
    device: str = "cuda"
    precision: str = "fp16"
    control_type: str = "canny"
    image_resolution: int = 512
    steps: int = 20
    guidance_scale: float = 9.0
    strength: float = 1.0
    guess_mode: bool = False
    eta: float = 0.0
    negative_prompt: str = (
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, low quality"
    )
    prompt_prefix: str = "best quality, extremely detailed"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    save_memory: bool = False


@dataclass
class Sam2Config:
    """Settings for SAM2 automatic mask generation.

    Attributes:
        repo_path: Path to the SAM2 submodule.
        model_cfg_path: Path to the SAM2 config yaml.
        checkpoint_path: Path to the SAM2 checkpoint.
        device: Torch device string.
        points_per_side: AMG points per side.
        pred_iou_thresh: AMG IoU threshold.
        stability_score_thresh: AMG stability threshold.
        min_mask_region_area: Minimum mask size.
    """

    repo_path: str = "sam2"
    model_cfg_path: str = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint_path: str = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 256


@dataclass
class SemanticSamConfig:
    """Settings for Semantic-SAM segmentation."""

    repo_path: str = "Semantic-SAM"
    config_path: str = "Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml"
    checkpoint_path: str = "model_Seg/swint_only_sam_many2many.pth"
    model_type: Optional[str] = "T"
    device: str = "cuda"
    points_per_side: int = 32
    points_per_batch: int = 200
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 10
    levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    resize_short_edge: Optional[int] = None


@dataclass
class DinoV2Config:
    """Settings for DINOv2 feature extraction."""

    model_name: str = "facebook/dinov2-base"
    checkpoint_path: Optional[str] = "model_Seg/dinov2_vitl14_pretrain.pth"
    repo_path: Optional[str] = None
    arch: str = "dinov2_vitl14"
    input_size: int = 518
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = True


@dataclass
class Blip2Config:
    """Settings for BLIP-2 captioning.

    Attributes:
        model_name: Hugging Face model id.
        device: Torch device string.
        dtype: Torch dtype name (float16/float32/bfloat16).
        max_new_tokens: Maximum tokens to generate.
        prompt: Optional text prompt for the captioner.
        local_files_only: Whether to force loading from local files only.
    """

    model_name: str = "Salesforce/blip2-flan-t5-xl"
    device: str = "cuda"
    dtype: str = "float16"
    max_new_tokens: int = 30
    prompt: Optional[str] = None
    local_files_only: bool = True


@dataclass
class ImaginationConfig:
    """Settings for the imagination module.

    Attributes:
        captioner_name: Identifier of the text captioning model used to
            describe the grayscale input.
        generator_name: Identifier of the diffusion or GAN backbone used to
            synthesize colorful references.
        num_candidates: Number of colorful reference candidates to generate for
            each grayscale input.
        guidance_scale: Strength of classifier-free guidance (or analogous
            control parameter) applied during sampling.
        seed: Optional seed for deterministic sampling.
        controlnet: ControlNet inference configuration.
        prompt_template: Format string applied to the caption.
        negative_prompt: Optional override for the negative prompt.
        blip2: Optional BLIP-2 configuration for automatic captioning.
    """

    captioner_name: str = "blip-base"
    generator_name: str = "stable-diffusion-xl"
    num_candidates: int = 8
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    prompt_template: str = "{caption}"
    negative_prompt: Optional[str] = None
    blip2: Optional[Blip2Config] = None


@dataclass
class RefinementConfig:
    """Settings for reference refinement and composition.

    Attributes:
        mask_blur: Gaussian blur radius applied to soft masks to reduce seams.
        alignment_reg_weight: Regularization weight when estimating structural
            alignment (e.g., optical flow or deformable attention).
        clip_weight: Weight of semantic similarity when ranking segments.
        depth_weight: Weight of structural consistency (depth/edges) when
            ranking segments.
        max_segments: Maximum number of segments to compose into the final
            reference.
        segment_min_area: Minimum pixel area for segments to keep.
        sam2: SAM2 configuration for automatic mask generation.
        rank_weights: Weights for ranking terms (e.g., edge/clip).
        semantic_sam: Semantic-SAM configuration for semantic-aware segments.
        dino: DINOv2 configuration for feature matching.
        distance_metric: Distance metric in feature space.
        overlap_thresh: Overlap threshold for suppressing duplicate segments.
    """

    mask_blur: float = 1.5
    alignment_reg_weight: float = 0.1
    clip_weight: float = 0.5
    depth_weight: float = 0.5
    max_segments: int = 5
    segment_min_area: int = 256
    sam2: Sam2Config = field(default_factory=Sam2Config)
    rank_weights: Dict[str, float] = field(default_factory=lambda: {"edge": 0.5, "clip": 0.5})
    semantic_sam: SemanticSamConfig = field(default_factory=SemanticSamConfig)
    dino: DinoV2Config = field(default_factory=DinoV2Config)
    distance_metric: str = "cosine"
    overlap_thresh: float = 0.9


@dataclass
class ColorizationConfig:
    """Settings for the reference-guided colorization network.

    Attributes:
        mode: Which colorization backend to use.
        backbone: Identifier for the base UNet/Transformer backbone.
        feature_channels: Channel dimensions for multi-scale fusion.
        use_cross_attention: Whether to inject reference features via
            cross-attention (recommended by the paper for controllability).
        use_segmentation_prior: Whether to use instance-aware priors during
            decoding to preserve object boundaries.
        controlnet: ControlNet inference configuration.
        color_prompt_template: Format string for color prompt injection.
    """

    mode: str = "sd15_controlnet"
    backbone: str = "unet-base"
    feature_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_cross_attention: bool = True
    use_segmentation_prior: bool = True
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    color_prompt_template: str = "{caption}, colorized"


@dataclass
class PipelineConfig:
    """Top-level configuration for the Imagine-Colorization pipeline."""

    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    colorization: ColorizationConfig = field(default_factory=ColorizationConfig)
