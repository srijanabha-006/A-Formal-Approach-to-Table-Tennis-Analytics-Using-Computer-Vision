import os
import shutil
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from scipy.interpolate import make_interp_spline
from roboflow import Roboflow
from ultralytics import YOLO


# =====================================================================
# 1. PROJECT PATH CONFIGURATION
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "src" / "model"

PERSON_MODEL_NAME = "yolov8x.pt"
BALL_MODEL_PATH = MODEL_DIR / "ball_model.pt"

# Input / output paths
INPUT_VIDEO_PATH = BASE_DIR / "input" / "input_video.mp4"
OUTPUT_VIDEO_PATH = BASE_DIR / "output" / "final_output_trajectory.mp4"


# =====================================================================
# 2. ROBOFLOW DATASET DOWNLOAD CONFIGURATION
# =====================================================================

ROBOFLOW_WORKSPACE = "madianou-kqrfk"
ROBOFLOW_PROJECT = "table-tennis-ball-detection"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov8"


def download_dataset():
    """
    Download the YOLOv8 dataset from Roboflow.

    The API key is read from the ROBOFLOW_API_KEY
    environment variable instead of being stored in the code.
    """

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is not set."
        )

    print("Connecting to Roboflow...")

    rf = Roboflow(api_key=api_key)

    project = rf.workspace(
        ROBOFLOW_WORKSPACE
    ).project(
        ROBOFLOW_PROJECT
    )

    version = project.version(
        ROBOFLOW_VERSION
    )

    dataset = version.download(
        ROBOFLOW_FORMAT
    )

    print("Dataset download complete!")

    return dataset


# =====================================================================
# 3. YOLO TRAINING CONFIGURATION
# =====================================================================

# Base model used by the original train_ball.py
BASE_TRAINING_MODEL = "yolov8n.pt"

# Original training configuration
TRAINING_EPOCHS = 30
TRAINING_IMAGE_SIZE = 640
TRAINING_BATCH_SIZE = 16
TRAINING_RUN_NAME = "tt_ball_model"


def train_ball_model(dataset):
    """
    Train the custom table-tennis-ball detector using
    the dataset downloaded from Roboflow.
    """

    # -------------------------------------------------------------
    # Find the downloaded dataset
    # -------------------------------------------------------------

    dataset_dir = Path(dataset.location)

    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml was not found at:\n{yaml_path}"
        )

    print(f"Training dataset:\n{yaml_path}")

    # -------------------------------------------------------------
    # Load YOLOv8 Nano base model
    # -------------------------------------------------------------

    model = YOLO(BASE_TRAINING_MODEL)

    print("Starting fine-tuning process...")

    # -------------------------------------------------------------
    # Train the model
    # -------------------------------------------------------------

    model.train(
        data=str(yaml_path),
        epochs=TRAINING_EPOCHS,
        imgsz=TRAINING_IMAGE_SIZE,
        batch=TRAINING_BATCH_SIZE,
        name=TRAINING_RUN_NAME
    )

    # -------------------------------------------------------------
    # Locate the best trained model
    # -------------------------------------------------------------

    best_model_path = Path(model.trainer.best)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Trained best model was not found at:\n"
            f"{best_model_path}"
        )

    # -------------------------------------------------------------
    # Save the trained model inside the project
    #
    # src/
    # └── model/
    #     └── ball_model.pt
    # -------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_model_path = MODEL_DIR / "ball_model.pt"

    shutil.copy2(
        best_model_path,
        final_model_path
    )

    print(
        f"Training Complete!\n"
        f"Custom model saved to:\n"
        f"{final_model_path}"
    )


# =====================================================================
# 4. TRAJECTORY SMOOTHER & CURVE GENERATOR
# =====================================================================

class TrajectoryCurveProcessor:

    def __init__(self, max_points=20):
        self.points = deque(maxlen=max_points)
        self.max_points = max_points

    def add_point(self, pt):
        self.points.append(pt)

    def interpolate_missing(self):
        """
        Interpolates missing frames (None values)
        linearly across the stored trajectory history.
        """

        pts = list(self.points)

        if len(pts) < 3:
            return [
                p for p in pts
                if p is not None
            ]

        cleaned = []
        i = 0

        while i < len(pts):

            if pts[i] is not None:
                cleaned.append(pts[i])
                i += 1

            else:

                # Find bounds of missing gap
                start_idx = i - 1

                while (
                    i < len(pts)
                    and pts[i] is None
                ):
                    i += 1

                end_idx = i

                if (
                    start_idx >= 0
                    and end_idx < len(pts)
                    and pts[end_idx] is not None
                ):

                    # Linearly interpolate between
                    # the surrounding valid points
                    p_start = np.array(
                        pts[start_idx]
                    )

                    p_end = np.array(
                        pts[end_idx]
                    )

                    steps = end_idx - start_idx

                    for s in range(1, steps):

                        interp_pt = (
                            p_start
                            + (p_end - p_start)
                            * (s / steps)
                        )

                        cleaned.append(
                            (
                                int(interp_pt[0]),
                                int(interp_pt[1])
                            )
                        )

        return cleaned

    def generate_smooth_spline(
        self,
        frame,
        color=(0, 255, 255)
    ):
        """
        Fits a smooth cubic B-Spline to the
        detected trajectory coordinates.
        """

        pts = self.interpolate_missing()

        if len(pts) < 4:

            # Draw simple lines if there are
            # too few points for spline fitting
            for j in range(1, len(pts)):

                cv2.line(
                    frame,
                    pts[j - 1],
                    pts[j],
                    color,
                    2,
                    cv2.LINE_AA
                )

            return

        pts_arr = np.array(pts)

        x = pts_arr[:, 0]
        y = pts_arr[:, 1]

        # Parameterize points by cumulative distance
        # along the trajectory
        distances = np.cumsum(
            np.sqrt(
                np.diff(
                    x,
                    prepend=x[0]
                ) ** 2
                +
                np.diff(
                    y,
                    prepend=y[0]
                ) ** 2
            )
        )

        if (
            distances[-1] == 0
            or np.all(
                np.diff(distances) == 0
            )
        ):
            return

        try:

            # Fit cubic B-Spline
            spl_x = make_interp_spline(
                distances,
                x,
                k=min(
                    3,
                    len(pts) - 1
                )
            )

            spl_y = make_interp_spline(
                distances,
                y,
                k=min(
                    3,
                    len(pts) - 1
                )
            )

            # Generate fine interpolation steps
            # for a smooth trajectory curve
            dense_distances = np.linspace(
                distances[0],
                distances[-1],
                num=50
            )

            smooth_x = spl_x(
                dense_distances
            )

            smooth_y = spl_y(
                dense_distances
            )

            smooth_pts = np.vstack(
                (
                    smooth_x,
                    smooth_y
                )
            ).T.astype(
                np.int32
            )

            # Draw smooth curve with
            # gradually changing thickness
            num_dense = len(smooth_pts)

            for k in range(
                1,
                num_dense
            ):

                p1 = tuple(
                    smooth_pts[k - 1]
                )

                p2 = tuple(
                    smooth_pts[k]
                )

                thickness = max(
                    1,
                    int(
                        np.sqrt(
                            k / num_dense
                        ) * 4
                    )
                )

                cv2.line(
                    frame,
                    p1,
                    p2,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

        except Exception:

            # Fallback to direct connection
            # if spline fitting fails
            for j in range(1, len(pts)):

                cv2.line(
                    frame,
                    pts[j - 1],
                    pts[j],
                    color,
                    2,
                    cv2.LINE_AA
                )


# =====================================================================
# 5. BALL DETECTION CONFIGURATION
# =====================================================================

CONF_THRESHOLD = 0.30
MAX_PHYSICAL_DISP = 160.0


# =====================================================================
# 6. PLAYER / BYTE TRACK CONFIGURATION
# =====================================================================

PERSON_CONF_THRESHOLD = 0.35
TRACKER_CONFIG = "bytetrack.yaml"
PERSON_CLASS = [0]

TABLE_ZONE_X1 = 0.15
TABLE_ZONE_Y1 = 0.25
TABLE_ZONE_X2 = 0.85
TABLE_ZONE_Y2 = 0.90

BALL_PLAYER_DISTANCE = 130


# =====================================================================
# 7. MAIN VIDEO GENERATION PIPELINE
# =====================================================================

def generate_final_video():

    # -------------------------------------------------------------
    # Verify required model files
    # -------------------------------------------------------------

    if not BALL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Ball model not found:\n"
            f"{BALL_MODEL_PATH}"
        )

    # -------------------------------------------------------------
    # Verify input video
    # -------------------------------------------------------------

    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found:\n"
            f"{INPUT_VIDEO_PATH}"
        )

    # -------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------

    person_model = YOLO(PERSON_MODEL_NAME)
    

    ball_model = YOLO(
        str(BALL_MODEL_PATH)
    )

    # Convert paths to strings for OpenCV
    input_video_path = str(
        INPUT_VIDEO_PATH
    )

    output_video_path = str(
        OUTPUT_VIDEO_PATH
    )

    # Ensure output directory exists
    OUTPUT_VIDEO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        input_video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video:\n"
            f"{INPUT_VIDEO_PATH}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = int(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    curve_processor = TrajectoryCurveProcessor(
        max_points=18
    )

    table_zone = [
        int(width * TABLE_ZONE_X1),
        int(height * TABLE_ZONE_Y1),
        int(width * TABLE_ZONE_X2),
        int(height * TABLE_ZONE_Y2)
    ]

    # Player classification state
    person_centers = {}
    motion_history = {}
    ball_hits = {}
    smoothed_player_scores = {}

    SLIDING_WINDOW_FRAMES = int(
        fps * 3
    )

    print(
        "Rendering smoothed spline trajectory... "
        "Please wait."
    )

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # -------------------------------------------------------------
        # A. BALL DETECTION & SPLINE CURVE FITTING
        # -------------------------------------------------------------

        ball_results = ball_model.predict(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        accepted_ball = None

        if (
            ball_results[0].boxes is not None
            and len(
                ball_results[0].boxes
            ) > 0
        ):

            boxes = (
                ball_results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            confs = (
                ball_results[0]
                .boxes
                .conf
                .cpu()
                .numpy()
            )

            # Pick candidate that respects
            # physical velocity limits
            for box, conf in zip(
                boxes,
                confs
            ):

                bx1, by1, bx2, by2 = map(
                    int,
                    box
                )

                bcx, bcy = (
                    (bx1 + bx2) // 2,
                    (by1 + by2) // 2
                )

                valid_pts = (
                    curve_processor
                    .interpolate_missing()
                )

                if len(valid_pts) > 0:

                    last_cx, last_cy = (
                        valid_pts[-1]
                    )

                    disp = np.sqrt(
                        (bcx - last_cx) ** 2
                        +
                        (bcy - last_cy) ** 2
                    )

                    if disp <= MAX_PHYSICAL_DISP:

                        accepted_ball = (
                            bcx,
                            bcy,
                            bx1,
                            by1,
                            bx2,
                            by2
                        )

                        break

                else:

                    accepted_ball = (
                        bcx,
                        bcy,
                        bx1,
                        by1,
                        bx2,
                        by2
                    )

                    break

        if accepted_ball is not None:

            bcx, bcy, bx1, by1, bx2, by2 = (
                accepted_ball
            )

            curve_processor.add_point(
                (bcx, bcy)
            )

            # Draw yellow bounding box
            # around active ball
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 255),
                2
            )

        else:

            # Append None so interpolation
            # can bridge the missing frame gap
            curve_processor.add_point(
                None
            )

        # Render smooth trajectory curve
        curve_processor.generate_smooth_spline(
            frame,
            color=(0, 255, 255)
        )

        # -------------------------------------------------------------
        # B. PLAYER SCORE & CLASSIFICATION (BYTE TRACK)
        # -------------------------------------------------------------

        person_results = person_model.track(
            frame,
            persist=True,
            classes=PERSON_CLASS,
            conf=PERSON_CONF_THRESHOLD,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        current_frame_tracked_persons = []

        valid_ball_pts = (
            curve_processor
            .interpolate_missing()
        )

        current_ball_center = (
            valid_ball_pts[-1]
            if len(valid_ball_pts) > 0
            else None
        )

        if (
            person_results[0].boxes is not None
            and person_results[0].boxes.id is not None
        ):

            boxes = (
                person_results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                person_results[0]
                .boxes
                .id
                .cpu()
                .numpy()
            )

            for box, track_id in zip(
                boxes,
                track_ids
            ):

                track_id = int(track_id)

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                cx, cy = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                # -------------------------------------------------
                # Court Score
                # -------------------------------------------------

                in_court = (
                    table_zone[0] <= cx <= table_zone[2]
                    and
                    table_zone[1] <= cy <= table_zone[3]
                )

                court_score = (
                    1.0
                    if in_court
                    else 0.0
                )

                # -------------------------------------------------
                # Motion Score
                # -------------------------------------------------

                if track_id not in motion_history:

                    motion_history[track_id] = deque(
                        maxlen=SLIDING_WINDOW_FRAMES
                    )

                    person_centers[track_id] = (
                        cx,
                        cy
                    )

                    disp = 0.0

                else:

                    prev_cx, prev_cy = (
                        person_centers[track_id]
                    )

                    disp = np.sqrt(
                        (cx - prev_cx) ** 2
                        +
                        (cy - prev_cy) ** 2
                    )

                    person_centers[track_id] = (
                        cx,
                        cy
                    )

                motion_history[
                    track_id
                ].append(disp)

                accumulated_motion = sum(
                    motion_history[
                        track_id
                    ]
                )

                # -------------------------------------------------
                # Ball Score
                # -------------------------------------------------

                if track_id not in ball_hits:
                    ball_hits[track_id] = 0

                if current_ball_center is not None:

                    dist_to_ball = np.sqrt(
                        (
                            cx
                            - current_ball_center[0]
                        ) ** 2
                        +
                        (
                            cy
                            - current_ball_center[1]
                        ) ** 2
                    )

                    if (
                        dist_to_ball
                        < BALL_PLAYER_DISTANCE
                    ):
                        ball_hits[
                            track_id
                        ] += 1

                current_frame_tracked_persons.append({
                    "track_id": track_id,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "court_score": court_score,
                    "accumulated_motion": accumulated_motion
                })

        # -------------------------------------------------------------
        # C. NORMALIZED & EXPONENTIALLY SMOOTHED PLAYER SCORES
        # -------------------------------------------------------------

        max_hits = (
            max(
                ball_hits.values()
            )
            if (
                len(ball_hits) > 0
                and
                max(
                    ball_hits.values()
                ) > 0
            )
            else 1.0
        )

        max_motion = (
            max(
                [
                    p[
                        "accumulated_motion"
                    ]
                    for p
                    in current_frame_tracked_persons
                ]
            )
            if (
                len(
                    current_frame_tracked_persons
                ) > 0
                and
                max(
                    [
                        p[
                            "accumulated_motion"
                        ]
                        for p
                        in current_frame_tracked_persons
                    ]
                ) > 0
            )
            else 1.0
        )

        eval_results = []

        for p in (
            current_frame_tracked_persons
        ):

            t_id = p["track_id"]

            c_score = p[
                "court_score"
            ]

            b_score = (
                ball_hits[t_id]
                / max_hits
            )

            m_score = (
                p["accumulated_motion"]
                / max_motion
            )

            raw_score = (
                (0.30 * c_score)
                +
                (0.50 * b_score)
                +
                (0.20 * m_score)
            )

            if (
                t_id
                not in smoothed_player_scores
            ):

                smoothed_player_scores[
                    t_id
                ] = raw_score

            else:

                smoothed_player_scores[
                    t_id
                ] = (
                    (
                        0.9
                        *
                        smoothed_player_scores[
                            t_id
                        ]
                    )
                    +
                    (
                        0.1
                        *
                        raw_score
                    )
                )

            eval_results.append({
                "track_id": t_id,
                "box": p["box"],
                "score": smoothed_player_scores[
                    t_id
                ]
            })

        eval_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # -------------------------------------------------------------
        # D. LABEL TOP 2 AS PLAYER, REST AS REFEREE
        # -------------------------------------------------------------

        for idx, item in enumerate(
            eval_results
        ):

            x1, y1, x2, y2 = item[
                "box"
            ]

            if idx < 2:

                label = (
                    f"Player "
                    f"(ID:{item['track_id']})"
                )

                color = (
                    0,
                    255,
                    0
                )

            else:

                label = (
                    f"Referee "
                    f"(ID:{item['track_id']})"
                )

                color = (
                    255,
                    128,
                    0
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        15,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # -------------------------------------------------------------
        # E. OUTPUT GENERATION
        # -------------------------------------------------------------

        out.write(frame)

    cap.release()
    out.release()

    print(
        f"SUCCESS! Broadcast-quality trajectory "
        f"video saved to:\n"
        f"{output_video_path}"
    )


# =====================================================================
# 8. ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    # The three functions represent the three stages
    # of the complete project:
    #
    # 1. download_dataset()
    #       -> Download dataset from Roboflow
    #
    # 2. train_ball_model(dataset)
    #       -> Train custom YOLOv8 ball detector
    #
    # 3. generate_final_video()
    #       -> Detect ball, track players and
    #          generate trajectory video
    #
    # The default execution runs only the final
    # video analysis using the already-trained model.

    generate_final_video()