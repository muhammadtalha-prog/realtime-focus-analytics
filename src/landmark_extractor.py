"""
landmark_extractor.py
---------------------
Wraps MediaPipe Face Mesh and Pose (Tasks API) into a single class.
Exposes per-frame landmark extraction for the feature engineering pipeline.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── Model download URLs ─────────────────────────────────────────────────────
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FACE_MODEL_FILE = "face_landmarker.task"
POSE_MODEL_FILE = "pose_landmarker_lite.task"


def _download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        print(f"Downloading model: {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  [OK] Downloaded {path}")


class LandmarkExtractor:
    """
    Extracts face and pose landmarks from a BGR camera frame.

    Returns a dict:
        {
          'face': list of 478 NormalizedLandmark objects  (or None)
          'pose': list of 33  NormalizedLandmark objects  (or None)
        }
    """

    def __init__(self):
        _download_if_missing(FACE_MODEL_URL, FACE_MODEL_FILE)
        _download_if_missing(POSE_MODEL_URL, POSE_MODEL_FILE)

        # ── Face Landmarker (478 pts including iris) ─────────────────────────
        face_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_FILE),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

        # ── Pose Landmarker (33 pts) ──────────────────────────────────────────
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_FILE),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

        print("LandmarkExtractor ready.")

    def extract(self, bgr_frame: np.ndarray) -> dict:
        """
        Args:
            bgr_frame: OpenCV BGR image (H, W, 3)
        Returns:
            dict with keys 'face' and 'pose'; values are landmark lists or None.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect(mp_img)
        pose_result = self.pose_landmarker.detect(mp_img)

        face_lms = face_result.face_landmarks[0] if face_result.face_landmarks else None
        pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        return {"face": face_lms, "pose": pose_lms}
