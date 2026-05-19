import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import yolo_data_split


class TestYoloDataSplitHelpers(unittest.TestCase):
    def test_parse_image_exts(self):
        exts = yolo_data_split._parse_image_exts("jpg, .png ,jpeg")
        self.assertEqual(exts, (".jpg", ".png", ".jpeg"))

    def test_main_uses_pipeline_style_arguments(self):
        with patch.object(yolo_data_split, "split_yolo_dataset", return_value={}) as mock_split:
            rc = yolo_data_split.main(
                [
                    "--source_root", "dataset",
                    "--output_root", "split_out",
                    "--train_ratio", "0.7",
                    "--val_ratio", "0.2",
                    "--test_ratio", "0.1",
                    "--seed", "11",
                    "--image_exts", ".jpg,.png",
                    "--use_stratified", "false",
                    "--rare_threshold", "9",
                ]
            )

        self.assertEqual(rc, 0)
        mock_split.assert_called_once()
        kwargs = mock_split.call_args.kwargs
        self.assertEqual(kwargs["train_ratio"], 0.7)
        self.assertEqual(kwargs["val_ratio"], 0.2)
        self.assertEqual(kwargs["test_ratio"], 0.1)
        self.assertEqual(kwargs["seed"], 11)
        self.assertEqual(kwargs["image_exts"], (".jpg", ".png"))
        self.assertFalse(kwargs["use_stratified"])
        self.assertEqual(kwargs["rare_threshold"], 9)


class TestYoloDataSplitIntegration(unittest.TestCase):
    def _build_tiny_yolo_dataset(self, root: Path, n_images: int = 6):
        images_dir = root / "images" / "session_1"
        labels_dir = root / "labels" / "session_1"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_images):
            img = np.zeros((24, 24, 3), dtype=np.uint8)
            cv2.rectangle(img, (4, 4), (18, 18), (255, 255, 255), thickness=-1)
            img_path = images_dir / f"sample_{idx}.jpg"
            lbl_path = labels_dir / f"sample_{idx}.txt"
            cv2.imwrite(str(img_path), img)
            class_id = idx % 2
            lbl_path.write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    def test_split_yolo_dataset_creates_train_val_test_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = tmp_path / "source"
            output_root = tmp_path / "output"
            self._build_tiny_yolo_dataset(source_root, n_images=6)

            splits = yolo_data_split.split_yolo_dataset(
                source_root=source_root,
                output_root=output_root,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=42,
                use_stratified=False,
            )

            self.assertEqual(sum(len(v) for v in splits.values()), 6)
            self.assertTrue((output_root / "images" / "train").exists())
            self.assertTrue((output_root / "images" / "val").exists())
            self.assertTrue((output_root / "images" / "test").exists())
            self.assertTrue((output_root / "labels" / "train").exists())
            self.assertTrue((output_root / "labels" / "val").exists())
            self.assertTrue((output_root / "labels" / "test").exists())

            copied_images = list((output_root / "images").rglob("*.jpg"))
            copied_labels = list((output_root / "labels").rglob("*.txt"))
            self.assertEqual(len(copied_images), 6)
            self.assertEqual(len(copied_labels), 6)


if __name__ == "__main__":
    unittest.main()

