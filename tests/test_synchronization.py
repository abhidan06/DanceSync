import unittest
import numpy as np
from synchronization import frame_disagreement, normalize_pose


def sample_pose():
    pose = np.zeros((33, 3))
    pose[:, 2] = 1
    pose[11, :2], pose[12, :2] = [-1, -2], [1, -2]
    pose[23, :2], pose[24, :2] = [-1, 0], [1, 0]
    return pose


class ScoringTests(unittest.TestCase):
    def test_translation_and_scale(self):
        a = sample_pose()
        b = a.copy()
        b[:, :2] = b[:, :2] * 3 + [100, 200]
        score, count, pairs = frame_disagreement([a, b])
        self.assertAlmostEqual(score, 0)
        self.assertEqual((count, pairs), (2, 1))

    def test_joint_change(self):
        a, b = sample_pose(), sample_pose()
        b[15, 0] += 2
        self.assertAlmostEqual(frame_disagreement([a, b])[0], 1 / 12)

    def test_missing_data(self):
        a = sample_pose()
        self.assertTrue(np.isnan(frame_disagreement([a])[0]))
        a[23, 2] = 0
        self.assertIsNone(normalize_pose(a))
        a = sample_pose()
        a[15, 0] = np.nan
        self.assertAlmostEqual(frame_disagreement([a, sample_pose()])[0], 0)

    def test_degenerate_and_insufficient_joints(self):
        a = sample_pose()
        a[[11, 12], :2] = a[[23, 24], :2]
        self.assertIsNone(normalize_pose(a))
        a = sample_pose()
        a[[13, 14, 15, 16, 25, 26, 27, 28], 2] = 0
        self.assertTrue(np.isnan(frame_disagreement([a, sample_pose()])[0]))


if __name__ == "__main__":
    unittest.main()
