import argparse
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imagine_colorization.adapters.blip2_adapter import Blip2Captioner
from imagine_colorization.config import Blip2Config


def _load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="测试 BLIP-2 图片描述能力")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--blip2-model", required=True, help="BLIP-2 模型名称或本地路径")
    parser.add_argument("--blip2-device", default="cuda", help="BLIP-2 设备")
    parser.add_argument("--blip2-dtype", default="float16", help="BLIP-2 dtype")
    parser.add_argument("--blip2-max-new-tokens", type=int, default=30, help="BLIP-2 生成长度")
    parser.add_argument(
        "--blip2-prompt",
        default="Question: Describe the image without mentioning colors. Answer:",
        help="BLIP-2 提示词",
    )
    parser.add_argument("--blip2-local-files-only", action="store_true", help="仅从本地加载 BLIP-2")
    args = parser.parse_args(argv)

    image = _load_image(args.image)
    config = Blip2Config(
        model_name=args.blip2_model,
        device=args.blip2_device,
        dtype=args.blip2_dtype,
        max_new_tokens=args.blip2_max_new_tokens,
        prompt=args.blip2_prompt,
        local_files_only=args.blip2_local_files_only,
    )
    captioner = Blip2Captioner(config)
    caption = captioner.generate(image).strip()
    if not caption:
        raise RuntimeError("BLIP-2 输出为空，请检查模型或提示词。")
    print(caption)


if __name__ == "__main__":
    main()
