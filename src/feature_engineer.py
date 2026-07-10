"""
feature_engineer.py
--------------------
Computes biometric features from raw MediaPipe landmarks each frame.

Features produced:
  - ear        : Eye Aspect Ratio (both eyes averaged)
  - mar        : Mouth Aspect Ratio (yawn detector)
  - gaze_x/y   : Estimated gaze deviation from center (normalised -1 to +1)
  - yaw/pitch  : Head pose angles in degrees (via solvePnP)
  - roll       : Head tilt angle in degrees
  - posture    : Shoulder-to-ear slouch angle in degrees (from pose landmarks)
"""

import cv2
import numpy as np
import math


# ─── Landmark indices for MediaPipe Face Mesh 478-pt model ──────────────────
# Left eye (from the subject's POV — right in mirror)
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE  = [33, 160, 158, 133, 153, 144]
# Left iris centre (MediaPipe 478-pt with iris)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# Mouth
MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]

# 6-point 3-D face model for solvePnP head-pose estimation
# Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
FACE_3D_MODEL = np.array([
    [0.0,      0.0,      0.0],       # Nose tip
    [0.0,     -330.0,   -65.0],      # Chin
    [-225.0,   170.0,  -135.0],      # Left eye corner
    [225.0,    170.0,  -135.0],      # Right eye corner
    [-150.0,  -150.0,  -125.0],      # Left mouth corner
    [150.0,   -150.0,  -125.0],      # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_2D_IDX = [1, 152, 226, 446, 57, 287]


class FeatureEngineer:
    """
    Stateless feature extractor — call compute_features(landmarks, frame_shape)
    each frame.
    """

    @staticmethod
    def _eye_aspect_ratio(eye_pts) -> float:
        """EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)"""
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    @staticmethod
    def _mouth_aspect_ratio(mouth_pts) -> float:
        """Simple vertical/horizontal ratio for yawn detection."""
        vert = np.linalg.norm(mouth_pts[2] - mouth_pts[6])
        horiz = np.linalg.norm(mouth_pts[0] - mouth_pts[4])
        return vert / (horiz + 1e-6)

    @staticmethod
    def _landmarks_to_px(landmarks, h, w) -> np.ndarray:
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    def compute_features(self, landmarks: dict, frame_shape: tuple) -> dict | None:
        """
        Args:
            landmarks: dict from LandmarkExtractor.extract()
            frame_shape: (H, W, C) tuple from the frame
        Returns:
            dict of features, or None if face not visible
        """
        face = landmarks.get("face")
        pose = landmarks.get("pose")

        if face is None:
            return None

        h, w = frame_shape[:2]

        # ── Convert face landmarks to pixel coords ────────────────────────────
        face_px = self._landmarks_to_px(face, h, w)

        # ── Eye Aspect Ratio ──────────────────────────────────────────────────
        left_eye_pts  = face_px[LEFT_EYE]
        right_eye_pts = face_px[RIGHT_EYE]
        ear = (self._eye_aspect_ratio(left_eye_pts) +
               self._eye_aspect_ratio(right_eye_pts)) / 2.0

        # ── Mouth Aspect Ratio ────────────────────────────────────────────────
        mouth_pts = face_px[MOUTH_OUTER]
        mar = self._mouth_aspect_ratio(mouth_pts)

        # ── Head Pose via solvePnP ────────────────────────────────────────────
        face_2d = face_px[FACE_2D_IDX].astype(np.float64)
        focal   = w                          # simple approximation
        cam_mat = np.array([
            [focal, 0,      w / 2],
            [0,     focal,  h / 2],
            [0,     0,      1   ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        yaw = pitch = roll = 0.0
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            # Euler angles from rotation matrix
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                roll  = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
                pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
                yaw   = math.degrees(math.atan2( rmat[1, 0], rmat[0, 0]))
            else:
                roll  = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
                pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
                yaw   = 0.0

        # ── Gaze estimation: iris centre deviation from eye centre ────────────
        # Use 478-pt iris landmarks (indices 469-472 right, 474-477 left)
        try:
            right_iris_px = face_px[RIGHT_IRIS]
            left_iris_px  = face_px[LEFT_IRIS]
            right_eye_centre = face_px[[33, 133]].mean(axis=0)
            left_eye_centre  = face_px[[362, 263]].mean(axis=0)
            right_iris_centre = right_iris_px.mean(axis=0)
            left_iris_centre  = left_iris_px.mean(axis=0)
            # Normalise by eye width
            r_eye_w = max(np.linalg.norm(face_px[133] - face_px[33]), 1.0)
            l_eye_w = max(np.linalg.norm(face_px[263] - face_px[362]), 1.0)
            gaze_x = ((right_iris_centre[0] - right_eye_centre[0]) / r_eye_w +
                      (left_iris_centre[0]  - left_eye_centre[0])  / l_eye_w) / 2.0
            gaze_y = ((right_iris_centre[1] - right_eye_centre[1]) / r_eye_w +
                      (left_iris_centre[1]  - left_eye_centre[1])  / l_eye_w) / 2.0
        except (IndexError, AttributeError):
            gaze_x, gaze_y = 0.0, 0.0

        # ── Posture: shoulder–ear slouch angle ────────────────────────────────
        posture_angle = 0.0
        if pose is not None:
            try:
                # MediaPipe Pose indices: 11=L shoulder, 12=R shoulder, 0=nose/head
                l_shoulder = np.array([pose[11].x * w, pose[11].y * h])
                r_shoulder = np.array([pose[12].x * w, pose[12].y * h])
                l_ear_lm   = np.array([pose[7].x * w,  pose[7].y * h])
                r_ear_lm   = np.array([pose[8].x * w,  pose[8].y * h])
                shoulder_mid = (l_shoulder + r_shoulder) / 2.0
                ear_mid      = (l_ear_lm  + r_ear_lm)   / 2.0
                vec = ear_mid - shoulder_mid
                # Angle from vertical (positive = leaning forward)
                posture_angle = math.degrees(
                    math.atan2(abs(vec[0]), abs(vec[1]))
                )
            except (IndexError, AttributeError):
                posture_angle = 0.0

        return {
            "ear":     round(float(ear), 4),
            "mar":     round(float(mar), 4),
            "gaze_x":  round(float(gaze_x), 4),
            "gaze_y":  round(float(gaze_y), 4),
            "yaw":     round(float(yaw), 2),
            "pitch":   round(float(pitch), 2),
            "roll":    round(float(roll), 2),
            "posture": round(float(posture_angle), 2),
        }
