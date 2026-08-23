# Table Tennis Ball Trajectory & Player Tracking

Computer vision pipeline that detects a table tennis ball in video, fits a smooth
spline trajectory to its motion and uses YOLOv8 + ByteTrack to classify people
in frame as **Player** or **Referee** based on court position, movement and
proximity to ball hits.

## Features

- **Ball detection** with a custom-trained YOLOv8 model, gated by a confidence
  threshold and a physical-displacement check (rejects impossible jumps between
  frames).
- **Missing-frame interpolation** — bridges frames where the ball wasn't
  detected by linearly interpolating between the nearest valid points.
- **Smooth trajectory rendering** using cubic B-spline fitting (`scipy`) with a
  fading trail thickness, falling back to straight lines when too few points
  are available.
- **Player/Referee classification** via a pretrained YOLOv8x person detector
  (fetched automatically by Ultralytics on first run) combined with ByteTrack
  multi-object tracking. Each tracked person is scored on:
  - Court-zone presence (30%)
  - Ball-proximity hit count (50%)
  - Accumulated motion over a rolling window (20%)
  - Scores are exponentially smoothed frame-to-frame; the top 2 tracked people
    are labeled **Player**, the rest **Referee**.
- **Dataset download & model training** helpers to fetch the ball dataset from
  Roboflow and fine-tune a YOLOv8n model on it.

## Project Structure

```
TABLE TENNIS PROJECT/
├── src/
│   ├── main.py
│   └── model/
│       └── ball_model.pt     # custom-trained ball detector (not tracked in git)
├── input/
│   └── input_video.mp4       # not tracked in git
├── output/
│   └── final_output_trajectory.mp4   # generated, not tracked in git
├── requirements.txt
└── README.md
```

> `ball_model.pt` and video files are excluded from version control — see
> `.gitignore` recommendations below. The YOLOv8x person-detection weights are
> **not** stored in the repo either; they are downloaded automatically by
> Ultralytics the first time the script runs (see Setup, step 4).

## Setup

1. Clone the repo and move into the project root.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set your Roboflow API key as an environment variable (required only for
   `download_dataset()` / training — not needed to just run inference):

   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here     # Windows: set ROBOFLOW_API_KEY=your_api_key_here
   ```

4. Place required files:
   - `src/model/ball_model.pt` — trained ball detector (see Training below)
   - `input/input_video.mp4` — the source video to analyze

   The person-detection model (`yolov8x.pt`) does **not** need to be added
   manually. It is downloaded automatically by Ultralytics the first time
   `generate_final_video()` runs and cached locally for subsequent runs.
   An internet connection is required for this first run.

## Usage

The pipeline has three independent stages, all defined in `src/main.py`:

### 1. Download the training dataset (optional, for retraining the ball model)

```python
from main import download_dataset
dataset = download_dataset()
```

Pulls the `table-tennis-ball-detection` dataset (v1, YOLOv8 format) from the
`madianou-kqrfk` Roboflow workspace. Requires `ROBOFLOW_API_KEY`.

### 2. Train the ball detector (optional)

```python
from main import download_dataset, train_ball_model

dataset = download_dataset()
train_ball_model(dataset)
```

Fine-tunes `yolov8n.pt` on the downloaded dataset (30 epochs, imgsz 640,
batch 16) and copies the best checkpoint to `src/model/ball_model.pt`.

### 3. Generate the annotated output video (default behavior)

```bash
python src/main.py
```

Reads `input/input_video.mp4`, runs ball + player detection frame by frame,
and writes an annotated video with:
- Yellow bounding box + smoothed spline trail on the ball
- Green boxes labeled `Player (ID:n)` for the top 2 scored people
- Orange boxes labeled `Referee (ID:n)` for everyone else

to `output/final_output_trajectory.mp4`.

On first run, Ultralytics will automatically download the pretrained
`yolov8x.pt` weights used for person detection.

## Configuration

Key tunable constants live near the top of `src/main.py`:

| Constant | Default | Purpose |
|---|---|---|
| `CONF_THRESHOLD` | 0.30 | Minimum confidence to accept a ball detection |
| `MAX_PHYSICAL_DISP` | 160.0 | Max allowed pixel displacement between frames (velocity gate) |
| `PERSON_CONF_THRESHOLD` | 0.35 | Minimum confidence for person detection |
| `TABLE_ZONE_X1/Y1/X2/Y2` | 0.15 / 0.25 / 0.85 / 0.90 | Normalized court bounding box used for the court score |
| `BALL_PLAYER_DISTANCE` | 130 | Pixel distance under which a player is credited with a "ball hit" |
| `TRAINING_EPOCHS` / `TRAINING_IMAGE_SIZE` / `TRAINING_BATCH_SIZE` | 30 / 640 / 16 | Ball model training hyperparameters |

## Requirements

See `requirements.txt`. Notable dependencies:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection & ByteTrack tracking
- [OpenCV](https://opencv.org/) — video I/O and drawing
- [SciPy](https://scipy.org/) — B-spline trajectory fitting
- [Roboflow](https://roboflow.com/) — dataset management

## Suggested `.gitignore`

```
*.pt
/input/
/output/
runs/
__pycache__/
venv/
.env
```

## Notes

- `ROBOFLOW_API_KEY` is read from the environment and is never stored in code
  — do not commit a `.env` file containing real credentials.
- The script validates that the ball model checkpoint and input video exist
  before running and raises descriptive errors if either is missing.
- `yolov8x.pt` is resolved by name rather than by local path, so Ultralytics
  handles downloading and caching it automatically — no need to manually
  source or upload this large file.
