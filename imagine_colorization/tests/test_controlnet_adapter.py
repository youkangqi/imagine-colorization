import unittest

import numpy as np

from imagine_colorization.adapters.controlnet_adapter import ControlNetAdapter
from imagine_colorization.config import ControlNetConfig


class _DummyDetector:
    def __init__(self, output: np.ndarray, return_tuple: bool = False) -> None:
        self.output = output
        self.return_tuple = return_tuple
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if self.return_tuple:
            return self.output, None
        return self.output


class ControlNetAdapterTest(unittest.TestCase):
    def _make_adapter(self, control_type: str) -> ControlNetAdapter:
        config = ControlNetConfig(control_type=control_type)
        adapter = ControlNetAdapter(config)
        adapter.load = lambda: None
        adapter._hwc3 = lambda x: x
        adapter._resize_image = lambda x, _: x
        return adapter

    def test_build_control_image_canny(self) -> None:
        adapter = self._make_adapter("canny")
        output = np.ones((2, 2, 3), dtype=np.uint8)
        detector = _DummyDetector(output)
        adapter._detector = detector

        result = adapter.build_control_image(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertTrue(np.array_equal(result, output))
        self.assertEqual(len(detector.calls), 1)
        _, low, high = detector.calls[0]
        self.assertEqual(low, adapter.config.canny_low_threshold)
        self.assertEqual(high, adapter.config.canny_high_threshold)

    def test_build_control_image_depth(self) -> None:
        adapter = self._make_adapter("depth")
        output = np.full((2, 2, 3), 7, dtype=np.uint8)
        detector = _DummyDetector(output, return_tuple=True)
        adapter._detector = detector

        result = adapter.build_control_image(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertTrue(np.array_equal(result, output))
        self.assertEqual(len(detector.calls), 1)


if __name__ == "__main__":
    unittest.main()
