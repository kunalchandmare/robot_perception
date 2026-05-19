import json
import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


@contextmanager
def _load_training_with_fakes():
    """Load training.py with temporary fake deps and restore sys.modules after use."""
    project_root = Path(__file__).resolve().parents[1]
    training_path = project_root / "training.py"

    saved_utils = sys.modules.get("utils")
    saved_ultralytics = sys.modules.get("ultralytics")

    fake_utils = types.ModuleType("utils")

    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    def print_gpu_info(_device):
        return None

    fake_utils.ensure_dir = ensure_dir
    fake_utils.print_gpu_info = print_gpu_info

    fake_ultralytics = types.ModuleType("ultralytics")

    class DummyYOLO:
        def __init__(self, model_name):
            self.model_name = model_name

        def train(self, **kwargs):
            return {"best": "weights/best.pt", "kwargs": kwargs}

        def val(self, **kwargs):
            return MagicMock(**{"kwargs": kwargs})

    fake_ultralytics.YOLO = DummyYOLO

    try:
        sys.modules["utils"] = fake_utils
        sys.modules["ultralytics"] = fake_ultralytics

        spec = importlib.util.spec_from_file_location("training_test_mod", training_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
        else:
            sys.modules.pop("utils", None)
        if saved_ultralytics is not None:
            sys.modules["ultralytics"] = saved_ultralytics
        else:
            sys.modules.pop("ultralytics", None)


class TestTrainingModelLoadingAndSaving(unittest.TestCase):
    def test_load_segmentation_model_uses_checkpoint_name(self):
        with _load_training_with_fakes() as training_mod, patch.object(training_mod, "YOLO") as mock_yolo:
            model = training_mod.load_segmentation_model("custom-seg.pt")

        mock_yolo.assert_called_once_with("custom-seg.pt")
        self.assertEqual(model, mock_yolo.return_value)

    def test_train_segmentation_model_forwards_save_related_args(self):
        fake_model = MagicMock()
        data_yaml = Path("data.yaml")
        project_dir = Path("output") / "runs"

        with _load_training_with_fakes() as training_mod:
            training_mod.train_segmentation_model(
                model=fake_model,
                data_yaml=data_yaml,
                project_dir=project_dir,
                run_name="demo_run",
                epochs=5,
                imgsz=320,
                batch=4,
                device=0,
                patience=3,
                workers=2,
            )

        fake_model.train.assert_called_once_with(
            data=str(data_yaml),
            task="segment",
            epochs=5,
            imgsz=320,
            batch=4,
            device=0,
            workers=2,
            patience=3,
            pretrained=True,
            amp=True,
            cache=False,
            project=str(project_dir),
            name="demo_run",
            exist_ok=True,
            plots=True,
        )

    def test_run_full_training_pipeline_finds_best_weights_under_run_folder(self):
        with _load_training_with_fakes() as training_mod, tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            dataset_root.mkdir()
            output_root = tmp_path / "output"
            output_root.mkdir()
            mapping_json = tmp_path / "class_id_to_name.json"
            mapping_json.write_text(json.dumps({"0": "chair"}), encoding="utf-8")

            best_weights = output_root / "runs" / "demo_run" / "weights" / "best.pt"
            best_weights.parent.mkdir(parents=True, exist_ok=True)
            best_weights.write_text("weights", encoding="utf-8")

            fake_model = MagicMock()
            fake_metrics = MagicMock()
            fake_metrics.box.map = 0.9
            fake_metrics.box.map50 = 0.8
            fake_metrics.box.map75 = 0.7
            fake_metrics.seg.map = 0.85
            fake_metrics.seg.map50 = 0.75
            fake_metrics.seg.map75 = 0.65
            summary = {"box_map50_95": {"value": 0.9}}
            results_df = pd.DataFrame({"epoch": [0], "metrics/precision": [0.5]})

            with patch.object(training_mod, "load_segmentation_model", return_value=fake_model) as mock_load_model, \
                 patch.object(training_mod, "print_gpu_info") as mock_gpu_info, \
                 patch.object(training_mod, "train_segmentation_model") as mock_train, \
                 patch.object(training_mod, "validate_on_test", return_value=fake_metrics) as mock_validate, \
                 patch.object(training_mod, "collect_metric_summary", return_value=summary) as mock_collect_summary, \
                 patch.object(training_mod, "save_metric_summary", return_value=output_root / "test_metrics.json") as mock_save_summary, \
                 patch.object(training_mod, "load_training_results_csv", return_value=(results_df, output_root / "runs" / "demo_run" / "results.csv")) as mock_load_csv, \
                 patch.object(training_mod, "plot_training_curves", return_value=output_root / "training_curves.png") as mock_plot:
                artifacts = training_mod.run_full_training_pipeline(
                    dataset_root=dataset_root,
                    output_root=output_root,
                    mapping_json=mapping_json,
                    model_name="custom-seg.pt",
                    run_name="demo_run",
                    epochs=5,
                    imgsz=320,
                    batch=4,
                    device=0,
                    workers=2,
                )

            mock_load_model.assert_called_once_with(model_name="custom-seg.pt")
            mock_gpu_info.assert_called_once_with(0)
            mock_train.assert_called_once()
            self.assertEqual(artifacts["best_model"], best_weights)
            self.assertTrue(best_weights.exists())
            self.assertEqual(artifacts["data_yaml"], output_root / "data.yaml")
            self.assertEqual(artifacts["class_names"], {0: "chair"})
            self.assertEqual(artifacts["test_metrics"], summary)
            mock_validate.assert_called_once()
            mock_collect_summary.assert_called_once_with(fake_metrics)
            mock_save_summary.assert_called_once_with(summary, output_root)
            mock_load_csv.assert_called_once_with(output_root / "runs", "demo_run")
            mock_plot.assert_called_once_with(results_df, output_root)



if __name__ == "__main__":
    unittest.main()

