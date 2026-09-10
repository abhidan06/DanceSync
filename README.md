# DanceSync

A small baseline for **same-time 2D pose agreement** among people performing the
same choreography. It uses pretrained models; it does not train a model or grade
dance quality. Lower disagreement means closer normalized poses, not a validated
synchronization percentage.

## Run

Use Python 3.10. From the repository directory in PowerShell, the supplied local
environment can be used with:

```powershell
.\venv\Scripts\python.exe main.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a new environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py data/raw/input_dance_video.mp4 --weights yolov8n.pt
```

The video and `yolov8n.pt` checkpoint are present locally but ignored by Git.
For another checkout, supply your own video and obtain the YOLOv8n detection
checkpoint from the Ultralytics project. The CLI requires an existing local
checkpoint; it does not silently download detector weights. MediaPipe's full pose
model is supplied by the pinned package. Dependency pins describe the supplied
environment, not a fully locked or independently verified fresh installation.

Optional flags: `--max-frames 60` processes a prefix for a smoke test;
`--output outputs/demo` chooses a new output directory. Existing output directories
are rejected to avoid overwriting a previous run.

Each run writes a silent `overlay.mp4`, `scores.csv`, and `disagreement.png` under
`outputs/<timestamp>/`. The CSV has zero-based frames, time in seconds, detection
count, extracted pose count, valid normalized pose count, valid pair count, and
disagreement. Missing scores are blank in CSV and gaps in the plot. Frame time is
frame index divided by source FPS, assuming constant-frame-rate video.

## Method and decisions

1. YOLOv8n detects people with confidence >= 0.5. This is person detection, not
   recognition of which people are dancers; use a clip without spectators.
2. Clamp each box to the image and run MediaPipe Pose on the clean crop. Static
   image mode prevents state from carrying between different people. Extract all
   poses before drawing. MediaPipe detection confidence is 0.5.
3. Convert crop landmarks to full-image pixels. Keep x, y and visibility. Use the
   shoulders, elbows, wrists, hips, knees and ankles (12 landmarks).
4. Subtract the midpoint of the hips and divide by the distance between hip and
   shoulder midpoints. This removes image translation and uniform scale, while
   keeping body orientation. Require both shoulders and hips to have visibility
   >= 0.5; reject nonfinite anchors or near-zero torso length.
5. For each unordered pair, average Euclidean distance over jointly reliable
   selected joints. Require at least six. Average these pair distances with equal
   pair weight to obtain the frame value.

For normalized landmarks q, the pair distance is
`d(a,b) = mean_j ||q[a,j] - q[b,j]||_2` over valid corresponding joints.
The units are torso lengths. Visibility and minimum-joint thresholds are explicit
heuristics, not experimentally optimized values. Visibility is a model estimate,
not a guarantee of landmark accuracy.

There is no identity tracking because this group statistic is invariant to the
order of detections within a frame. There is no audio processing because this
baseline does not measure alignment to music. There is no temporal warping or
smoothing; both would change the question or hide failures before a baseline is
understood. `synchronization.py` contains pure NumPy functions, `pose.py` handles
inference/drawing, and `main.py` handles the video and outputs.

## Limitations

- This is a proxy for agreement at the same instant, not a temporal lag estimator.
  Identical stationary poses can score zero; no claim of synchronized movement
  follows from that alone.
- Similar camera-facing directions and the same choreography are assumed.
  Perspective, occlusion, body proportions, turns and foreshortening still matter.
- Bad detections or crops can produce misleading poses. Multiple detections can
  overlap; there is no identity verification or guaranteed one pose per person.
- Changes in detected people or valid joints can change the value. Inspect the
  coverage columns and video alongside the curve. Missing data is not agreement.
- The group value cannot identify who is wrong. There is no labeled evaluation,
  accuracy result, synchronization threshold, or calibrated quality score.
- Frame decoding stops when OpenCV cannot read the next frame; a damaged file can
  therefore truncate processing. Check input/output frame counts when validating.
- The generated video omits audio.

## Validation

Unit tests cover translation/scale invariance, an analytically known joint
perturbation, missing data, unreliable anchors and degenerate torso geometry.
These are implementation sanity checks, not performance evaluation. A useful next
research step is to collect and manually inspect synchronized and deliberately
delayed examples before making claims about discrimination or timing.

### Local verification (2026-09-10)

The four unit tests passed in the supplied Python 3.10 environment. Running
`main.py --output outputs/verified_run` on the supplied video completed successfully.
The generated video was decoded back to 1,529 frames at 30 FPS, matching the input
frame count and 1,529 CSV rows. There were 1,528 frames with a computable score;
frame 131 had insufficient usable pose information. This is coverage, not accuracy.

The first, middle and last output frames were visually inspected, along with the
plot. Skeleton coordinates aligned with the image, but occlusions and missing poses
remain visible. Detection counts ranged from 3 to 12, and the curve contains sharp
spikes; their cause has not been labeled or evaluated. Do not interpret every peak
as a dancer making a mistake. A clean dependency installation has not been tested.
