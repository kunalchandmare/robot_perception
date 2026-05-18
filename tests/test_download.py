import importlib
import runpy
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


@contextmanager
def fake_download_env(expected_md5="ok"):
    """Provide isolated fake modules so download.py can be tested safely."""
    fake_utils = types.ModuleType("utils")
    fake_utils.align_all_masks = lambda masks, rgb_img: masks

    fake_rh = types.ModuleType("robotathome")

    class DummyLog:
        @staticmethod
        def set_log_level(_):
            return None

    class DummyLogger:
        @staticmethod
        def info(*_args, **_kwargs):
            return None

    class DummyRobotAtHome:
        def __init__(self, *args, **kwargs):
            pass

    fake_rh.log = DummyLog()
    fake_rh.logger = DummyLogger()
    fake_rh.RobotAtHome = DummyRobotAtHome
    fake_rh.get_labeled_img = lambda annotation, rgb_img: (annotation, None)
    fake_rh.download_calls = []
    fake_rh.uncompress_calls = []

    def _download(url, out_dir):
        fake_rh.download_calls.append((url, out_dir))

    def _uncompress(src, dst):
        fake_rh.uncompress_calls.append((src, dst))

    fake_rh.download = _download
    fake_rh.uncompress = _uncompress
    fake_rh.get_md5 = lambda _path: expected_md5

    with patch.dict(sys.modules, {"utils": fake_utils, "robotathome": fake_rh}):
        try:
            yield fake_rh
        finally:
            # Ensure later tests import a clean download module state.
            sys.modules.pop("download", None)


def import_download_with_fakes(expected_md5="ok"):
    """Import download.py under fake deps and return (module, fake_robotathome_module)."""
    cm = fake_download_env(expected_md5=expected_md5)
    fake_rh = cm.__enter__()
    try:
        sys.modules.pop("download", None)
        module = importlib.import_module("download")
        return module, fake_rh, cm
    except Exception:
        cm.__exit__(*sys.exc_info())
        raise


class TestDownloadCLI(unittest.TestCase):
    def test_cli_custom_args_work_with_small_accessible_url(self):
        small_accessible_url = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"

        with fake_download_env(expected_md5="ok") as fake_rh, TemporaryDirectory() as tmp:
            argv = [
                "download.py",
                "--out-dir",
                tmp,
                "--dataset-url",
                small_accessible_url,
                "--dataset-filename",
                "Python.gitignore",
                "--dataset-md5",
                "ok",
                "--no-force-download",
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module("download", run_name="__main__")

            self.assertEqual(len(fake_rh.download_calls), 1)
            self.assertEqual(fake_rh.download_calls[0][0], small_accessible_url)
            self.assertEqual(fake_rh.uncompress_calls, [])

    def test_cli_requires_url_filename_md5_together(self):
        with fake_download_env(expected_md5="ok"), TemporaryDirectory() as tmp:
            argv = ["download.py", "--out-dir", tmp, "--dataset-url", "https://example.com/file.txt"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as exc:
                    runpy.run_module("download", run_name="__main__")

        self.assertEqual(exc.exception.code, 2)


class TestDownloadFunction(unittest.TestCase):
    def test_download_rh_non_archive_skips_extract(self):
        small_accessible_url = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"

        download_mod, fake_rh, cm = import_download_with_fakes(expected_md5="abc123")
        try:
            with TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                file_path = out_dir / "Python.gitignore"
                file_path.write_text("dummy", encoding="utf-8")

                source_specs = [
                    {
                        "url": small_accessible_url,
                        "filename": "Python.gitignore",
                        "md5": "abc123",
                    }
                ]

                processed = download_mod.download_rh(
                    out_dir=out_dir,
                    force_download=True,
                    source_specs=source_specs,
                )

            self.assertEqual(processed, ["Python.gitignore"])
            self.assertEqual(len(fake_rh.download_calls), 1)
            self.assertEqual(fake_rh.download_calls[0][0], small_accessible_url)
            self.assertEqual(fake_rh.uncompress_calls, [])
            self.assertEqual(fake_rh.get_md5(str(file_path)), "abc123")
        finally:
            cm.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
