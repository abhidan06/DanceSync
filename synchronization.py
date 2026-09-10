"""Same-frame pose disagreement, measured in torso-length units."""
from itertools import combinations
import numpy as np

JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
ANCHORS = [11, 12, 23, 24]


def normalize_pose(landmarks, visibility_threshold=0.5):
    """Convert (33, 3) image x/y/visibility into normalized xy and validity."""
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (33, 3):
        raise ValueError("Expected 33 landmarks with x, y, visibility")
    valid = np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] >= visibility_threshold)
    if not valid[ANCHORS].all():
        return None
    hips = landmarks[[23, 24], :2].mean(axis=0)
    shoulders = landmarks[[11, 12], :2].mean(axis=0)
    scale = np.linalg.norm(shoulders - hips)
    if scale <= 1e-6:
        return None
    return (landmarks[:, :2] - hips) / scale, valid


def frame_disagreement(poses, visibility_threshold=0.5, min_joints=6):
    """Return distance (NaN if unavailable), valid pose count, valid pair count."""
    normalized = [normalize_pose(p, visibility_threshold) for p in poses]
    normalized = [p for p in normalized if p is not None]
    distances = []
    for (a, valid_a), (b, valid_b) in combinations(normalized, 2):
        joints = JOINTS[(valid_a & valid_b)[JOINTS]]
        if len(joints) >= min_joints:
            distances.append(float(np.linalg.norm(a[joints] - b[joints], axis=1).mean()))
    return (float(np.mean(distances)) if distances else float("nan"),
            len(normalized), len(distances))
