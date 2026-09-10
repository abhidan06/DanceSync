"""Check batch input selection and output separation without running inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main import video_jobs


class VideoJobsTests(unittest.TestCase):
    def test_directory_filters_and_sorts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["b.MP4", "a.mp4", "notes.txt"]:
                (root / name).touch()
            (root / "nested").mkdir()
            (root / "nested" / "hidden.mp4").touch()
            output = root / "output"
            self.assertEqual(video_jobs(root, output), [
                (root / "a.mp4", output / "a"),
                (root / "b.MP4", output / "b")])

    def test_single_file_keeps_output_path(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.touch()
            output = Path(directory) / "results"
            self.assertEqual(video_jobs(video, output), [(video, output)])

    def test_empty_or_missing_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for source in (root, root / "missing"):
                with self.assertRaises(ValueError):
                    video_jobs(source, root / "output")
