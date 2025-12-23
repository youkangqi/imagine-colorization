import unittest

import numpy as np

from imagine_colorization.config import ImaginationConfig
from imagine_colorization.imagination import ImaginationModule
from imagine_colorization.types import ColorizationSample


class _FakeControlNet:
    def __init__(self) -> None:
        self.generate_calls = []
        self.prepare_calls = 0
        self.build_calls = 0

    def prepare_control_image(self, image: np.ndarray) -> np.ndarray:
        self.prepare_calls += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def build_control_image(self, image: np.ndarray) -> np.ndarray:
        self.build_calls += 1
        return np.ones((4, 4, 3), dtype=np.uint8)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: np.ndarray,
        num_samples: int,
        seed: int | None,
        **kwargs,
    ) -> list[np.ndarray]:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "control_shape": control_image.shape,
                "num_samples": num_samples,
                "seed": seed,
            }
        )
        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(num_samples)]


class ImaginationModuleTest(unittest.TestCase):
    def test_imagination_generates_candidates(self) -> None:
        config = ImaginationConfig(
            num_candidates=2,
            prompt_template="Scene: {caption}",
            negative_prompt="bad",
        )
        module = ImaginationModule(config)
        module.controlnet = _FakeControlNet()
        sample = ColorizationSample(
            grayscale=np.zeros((4, 4), dtype=np.uint8),
            caption="test caption",
        )

        outputs = module(sample)

        self.assertEqual(outputs.caption, "test caption")
        self.assertEqual(len(outputs.candidates), 2)
        self.assertEqual(module.controlnet.build_calls, 1)
        self.assertEqual(module.controlnet.prepare_calls, 0)
        self.assertEqual(outputs.candidates[0].control_image.shape, (4, 4, 3))
        self.assertEqual(outputs.candidates[0].seed, config.seed)
        self.assertEqual(outputs.candidates[0].score, 1.0)

        call = module.controlnet.generate_calls[0]
        self.assertEqual(call["prompt"], "Scene: test caption")
        self.assertEqual(call["negative_prompt"], "bad")
        self.assertEqual(call["num_samples"], 2)

    def test_imagination_prefers_edges(self) -> None:
        config = ImaginationConfig(prompt_template="{caption}")
        module = ImaginationModule(config)
        module.controlnet = _FakeControlNet()
        sample = ColorizationSample(
            grayscale=np.zeros((4, 4), dtype=np.uint8),
            edges=np.zeros((4, 4, 3), dtype=np.uint8),
            caption="edge case",
        )

        _ = module(sample)

        self.assertEqual(module.controlnet.prepare_calls, 1)
        self.assertEqual(module.controlnet.build_calls, 0)


if __name__ == "__main__":
    unittest.main()
