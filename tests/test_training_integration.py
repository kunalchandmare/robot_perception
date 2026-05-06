import importlib.util
import unittest
from pathlib import Path


class TestTrainingIntegration(unittest.TestCase):
    def test_validate_on_test_with_real_artifacts(self):
        if importlib.util.find_spec("ultralytics") is None:
            self.skipTest("ultralytics is not installed in the active environment.")

        project_root = Path(__file__).resolve().parents[1]
        best_model_path = project_root / "output" / "runs" / "robotathome_seg_strat" / "weights" / "best.pt"
        data_yaml = project_root / "output" / "data.yaml"

        if not best_model_path.exists():
            self.skipTest(f"Missing trained weights: {best_model_path}")
        if not data_yaml.exists():
            self.skipTest(f"Missing dataset config: {data_yaml}")

        import training

        metrics = training.validate_on_test(
            model_path=best_model_path,
            data_yaml=data_yaml,
            imgsz=640,
            batch=32,
            device=0,
            workers=2,
        )

        self.assertIsNotNone(metrics)
        self.assertTrue(hasattr(metrics, "box"))
        self.assertTrue(hasattr(metrics, "seg"))
        self.assertGreaterEqual(float(metrics.box.map50), 0.0)
        self.assertLessEqual(float(metrics.box.map50), 1.0)
        self.assertGreaterEqual(float(metrics.box.map), 0.51)
        self.assertGreaterEqual(float(metrics.seg.map50), 0.0)
        self.assertLessEqual(float(metrics.seg.map50), 1.0)
        self.assertGreaterEqual(float(metrics.seg.map), 0.51)


if __name__ == "__main__":
    unittest.main()

