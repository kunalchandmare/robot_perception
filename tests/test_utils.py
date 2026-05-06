import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


if importlib.util.find_spec("cv2") is None:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.ROTATE_90_CLOCKWISE = 0
    fake_cv2.ROTATE_90_COUNTERCLOCKWISE = 1
    fake_cv2.INTER_NEAREST = 0
    fake_cv2.COLOR_BGR2RGB = 0

    def _rotate(arr, flag):
        if flag == fake_cv2.ROTATE_90_CLOCKWISE:
            return np.rot90(arr, k=3)
        return np.rot90(arr, k=1)

    def _resize(arr, size, interpolation=None):
        width, height = size
        out = np.zeros((height, width), dtype=arr.dtype)
        src_h, src_w = arr.shape[:2]
        copy_h = min(height, src_h)
        copy_w = min(width, src_w)
        out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return out

    def _imread(path):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def _cvt_color(arr, code):
        return arr[..., ::-1]

    fake_cv2.rotate = _rotate
    fake_cv2.resize = _resize
    fake_cv2.imread = _imread
    fake_cv2.cvtColor = _cvt_color
    sys.modules["cv2"] = fake_cv2

if importlib.util.find_spec("torch") is None:
    fake_torch = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def get_device_name(device_id):
            return "cpu"

        @staticmethod
        def memory_allocated(device_id):
            return 0

        @staticmethod
        def memory_reserved(device_id):
            return 0

    fake_torch.cuda = FakeCuda()
    sys.modules["torch"] = fake_torch

import utils


class MatplotlibCleanupTestCase(unittest.TestCase):
    def tearDown(self):
        plt.close("all")


class TestUtilsFilesystemAndJson(MatplotlibCleanupTestCase):
    def test_ensure_dir_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "nested" / "folder"
            utils.ensure_dir(target_dir)
            utils.ensure_dir(target_dir)
            self.assertTrue(target_dir.exists())
            self.assertTrue(target_dir.is_dir())

            json_path = target_dir / "sample.json"
            payload = {"name": "robot", "values": [1, 2, 3]}
            utils.save_json(payload, json_path)
            self.assertEqual(utils.load_json(json_path), payload)

    def test_save_json_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "deep" / "nested" / "config.json"
            utils.save_json({"enabled": True}, json_path)
            self.assertTrue(json_path.exists())
            self.assertEqual(utils.load_json(json_path), {"enabled": True})


class TestUtilsInteraction(MatplotlibCleanupTestCase):
    def test_ask_yes_no_respects_user_input_and_default(self):
        with patch("builtins.input", return_value="yes"):
            self.assertTrue(utils.ask_yes_no("Continue?", default=False))

        with patch("builtins.input", return_value=""):
            self.assertTrue(utils.ask_yes_no("Continue?", default=True))

        with patch("builtins.input", return_value="n"):
            self.assertFalse(utils.ask_yes_no("Continue?", default=True))


class TestUtilsMaskAlignment(MatplotlibCleanupTestCase):
    def test_align_all_masks_image_handles_matching_and_swapped_shapes(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        same_shape_mask = np.ones((4, 5), dtype=np.uint8)
        swapped_shape_mask = np.ones((5, 4), dtype=np.uint8)

        aligned = utils.align_all_masks_image([same_shape_mask, swapped_shape_mask], image)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0].shape, (4, 5))
        self.assertTrue(np.array_equal(aligned[0], same_shape_mask))
        self.assertEqual(aligned[1].shape, (4, 5))

    def test_align_all_masks_delegates_to_image_reader(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        masks = [np.ones((4, 5), dtype=np.uint8)]

        with patch.object(utils.cv2, "imread", return_value=image) as mock_imread:
            aligned = utils.align_all_masks(masks, "dummy.png")

        mock_imread.assert_called_once_with("dummy.png")
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].shape, (4, 5))

    def test_align_all_masks_image_resizes_non_swapped_mask(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        mismatched_mask = np.ones((2, 3), dtype=np.uint8)

        aligned = utils.align_all_masks_image([mismatched_mask], image)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].shape, (4, 5))


class TestUtilsPlotting(MatplotlibCleanupTestCase):
    def test_plot_helpers_return_axes(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        label_lines = ["0 0.2 0.2 0.8 0.2 0.8 0.8"]

        ax1 = utils.plot_image(image, title="img")
        ax2 = utils.plot_mask_overlay(image, [mask], title="mask")
        ax3 = utils.plot_yolo_bboxes(image, label_lines, title="bbox")

        self.assertEqual(ax1.get_title(), "img")
        self.assertEqual(ax2.get_title(), "mask")
        self.assertEqual(ax3.get_title(), "bbox")

    def test_plot_yolo_bboxes_ignores_short_invalid_lines(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        ax = utils.plot_yolo_bboxes(image, ["0 0.1 0.2"], title="invalid")
        self.assertEqual(ax.get_title(), "invalid")
        self.assertEqual(len(ax.patches), 0)


class TestUtilsGpuInfo(MatplotlibCleanupTestCase):
    def test_print_gpu_info_prints_expected_lines(self):
        with patch.object(utils.torch.cuda, "is_available", return_value=False), \
             patch.object(utils.torch.cuda, "device_count", return_value=0), \
             patch.object(utils.torch.cuda, "get_device_name", return_value="Mock GPU"), \
             patch.object(utils.torch.cuda, "memory_allocated", return_value=0), \
             patch.object(utils.torch.cuda, "memory_reserved", return_value=0), \
             patch("builtins.print") as mock_print:
            utils.print_gpu_info(0)

        printed = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(any("CUDA GPU not available. Using CPU." in line for line in printed))
        self.assertTrue(any("CUDA available: False" in line for line in printed))
        self.assertTrue(any("GPU name: Mock GPU" in line for line in printed))


if __name__ == "__main__":
    unittest.main()


