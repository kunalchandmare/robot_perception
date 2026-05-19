import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import cv2

# Force headless backend for visualization smoke tests
matplotlib.use("Agg")

import annotation_convert


class FolderBackedFakeDB:
    """Minimal RobotAtHome-like adapter backed by tests/data rgbd files."""

    def __init__(self, id_to_intensity_path):
        self._id_to_intensity_path = id_to_intensity_path

    def get_sensor_observations(self, _sensor_name="lblrgbd"):
        return pd.DataFrame({"id": sorted(self._id_to_intensity_path.keys())})

    def get_RGBD_files(self, obs_id):
        obs_id = int(obs_id)
        intensity_path = self._id_to_intensity_path[obs_id]
        depth_candidate = intensity_path.with_name(f"{obs_id}_depth.png")
        depth_path = str(depth_candidate) if depth_candidate.exists() else ""
        return str(intensity_path), depth_path

    def get_RGBD_labels(self, obs_id):
        obs_id = int(obs_id)
        intensity = cv2.imread(str(self._id_to_intensity_path[obs_id]), cv2.IMREAD_GRAYSCALE)
        if intensity is None:
            raise RuntimeError(f"Missing intensity image for obs_id={obs_id}")

        mask = (intensity > 0).astype(np.uint8) * 255
        if mask.sum() == 0:
            # Ensure at least one visible object region for polygon conversion.
            h, w = mask.shape
            y1, y2 = max(1, h // 4), max(2, h // 2)
            x1, x2 = max(1, w // 4), max(2, w // 2)
            mask[y1:y2, x1:x2] = 255

        return pd.DataFrame({"mask": [mask], "object_type_id": [1]})

    def id2name(self, class_id, _name_mode="ot"):
        mapping = {1: "chair", 2: "table", 3: "sofa"}
        return mapping.get(int(class_id), f"class_{class_id}")


class TestAnnotationConvertHelpers(unittest.TestCase):
    def test_prepare_binary_mask_and_polygon(self):
        mask = [[0, 255], [0, 0]]
        binary = annotation_convert.prepare_binary_mask(mask)
        self.assertEqual(binary.shape, (2, 2))
        self.assertEqual(int(binary.sum()), 1)

        # Simple rectangular mask should produce one valid YOLO polygon line
        rect = np.zeros((20, 20), dtype=np.uint8)
        rect[5:15, 5:15] = 255
        line = annotation_convert.mask_to_yolo_polygon(
            mask=rect,
            class_id=3,
            img_w=20,
            img_h=20,
            epsilon_ratio=0.001,
        )
        self.assertIsNotNone(line)
        self.assertTrue(line.startswith("3 "))


class TestAnnotationConvertRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        test_data_root = project_root / "tests" / "data"
        rgbd_path = test_data_root / "files" / "rgbd"
        scene_path = test_data_root / "files" / "scene"

        if not test_data_root.exists() or not rgbd_path.exists() or not scene_path.exists():
            raise unittest.SkipTest(
                "Missing test data folder. Expected structure: tests/data, tests/data/files/rgbd, tests/data/files/scene"
            )

        intensity_files = sorted(rgbd_path.rglob("*_intensity.png"))
        if len(intensity_files) < 2:
            raise unittest.SkipTest("Need at least two *_intensity.png files in tests/data/files/rgbd.")

        id_to_path = {}
        for p in intensity_files:
            prefix = p.stem.split("_")[0]
            if prefix.isdigit():
                id_to_path[int(prefix)] = p

        sample_ids = sorted(id_to_path.keys())[:2]
        if len(sample_ids) < 2:
            raise unittest.SkipTest("Could not infer two numeric observation IDs from intensity filenames.")

        cls.sample_ids = sample_ids
        cls.db = FolderBackedFakeDB({obs_id: id_to_path[obs_id] for obs_id in sample_ids})
        cls.rgbd_root = rgbd_path.resolve()

    def test_get_yolo_lines_for_observation_with_real_sample(self):
        for obs_id in self.sample_ids:
            image, rgb_path, lines = annotation_convert.get_yolo_lines_for_observation(
                self.db,
                obs_id,
                epsilon_ratio=0.002,
            )
            self.assertIsNotNone(image)
            self.assertIsNotNone(rgb_path)
            self.assertTrue(Path(rgb_path).exists())
            self.assertIsInstance(lines, list)

    def test_convert_df_to_yolo_seg_writes_outputs(self):
        # Use a temporary output folder and limit conversion to a few observations.
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            sampled_obs = self.db.get_sensor_observations("lblrgbd").head(2)
            with patch.object(self.db, "get_sensor_observations", return_value=sampled_obs):
                annotation_convert.convert_df_to_yolo_seg(
                    rh_db=self.db,
                    output_root=output_root,
                    rgbd_root=self.rgbd_root,
                    epsilon_ratio=0.002,
                )

            self.assertTrue((output_root / "images").exists())
            self.assertTrue((output_root / "labels").exists())

            image_files = list((output_root / "images").rglob("*.jpg"))
            label_files = list((output_root / "labels").rglob("*.txt"))

            self.assertGreater(len(image_files), 0)
            self.assertGreater(len(label_files), 0)

    def test_test_observation_visualization_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "visualization_smoke.png"

            def _capture_show():
                fig = annotation_convert.plt.gcf()
                fig.savefig(snapshot_path, dpi=100, bbox_inches="tight")

            with patch.object(annotation_convert.plt, "show", side_effect=_capture_show) as mock_show:
                annotation_convert.test_observation_visualization(
                    rh_db=self.db,
                    obs_id=self.sample_ids[0],
                    epsilon_ratio=0.002,
                )

            mock_show.assert_called_once()
            self.assertTrue(snapshot_path.exists())
            self.assertGreater(snapshot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
