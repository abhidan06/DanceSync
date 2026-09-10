"""Person detection and independent MediaPipe inference on clean crops."""
import cv2
import numpy as np


def extract_poses(frame, detector, estimator):
    height, width = frame.shape[:2]
    result = detector.predict(frame, classes=[0], conf=0.5, verbose=False)[0]
    boxes, poses = [], []
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1 = np.floor(box[:2]).astype(int)
        x2, y2 = np.ceil(box[2:]).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        prediction = estimator.process(crop)
        if prediction.pose_landmarks:
            poses.append(np.array([
                [x1 + lm.x * (x2 - x1), y1 + lm.y * (y2 - y1), lm.visibility]
                for lm in prediction.pose_landmarks.landmark
            ]))
    return boxes, poses


def draw_poses(frame, boxes, poses, connections, threshold=0.5):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    for landmarks in poses:
        valid = np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] >= threshold)
        for a, b in connections:
            if valid[a] and valid[b]:
                cv2.line(frame, tuple(landmarks[a, :2].astype(int)),
                         tuple(landmarks[b, :2].astype(int)), (0, 220, 0), 2)
        for x, y, visibility in landmarks[valid]:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
