"""
focus_scorer.py
---------------
Converts per-frame biometric features into a continuous focus score [0.0 – 1.0]
and a discrete state label, using a sliding-window heuristic approach.

Design rationale (for CV/interview discussions):
  - No labeled dataset required: proxy signals (EAR, gaze, head pose) act as
    weak labels for focus states
  - Interpretable and auditable: every contributor to the score is visible
  - Temporal smoothing via Exponential Moving Average prevents flickering
"""

from collections import deque
import numpy as np

# ─── Tunable thresholds ────────────────────────────────────────────────────
EAR_CLOSED_THRESH   = 0.20   # Below this → eye is closed
EAR_NORMAL_LOW      = 0.25   # Healthy blink range
EAR_NORMAL_HIGH     = 0.35
BLINK_RATE_MIN      = 5      # blinks/minute — below this → staring/strain
BLINK_RATE_MAX      = 20     # blinks/minute — above this → very drowsy
MAR_YAWN_THRESH     = 0.65   # MAR above this → yawning
GAZE_CENTRE_THRESH  = 0.20   # Iris deviation < this → looking at screen
HEAD_YAW_THRESH     = 20.0   # |yaw| above this → looking sideways
HEAD_PITCH_THRESH   = 20.0   # |pitch| above this → looking up/down
POSTURE_GOOD_THRESH = 15.0   # degrees of forward lean — below this → good posture
POSTURE_BAD_THRESH  = 35.0   # degrees — above this → slouching badly

# EMA smoothing factor (higher = more reactive, lower = more stable)
EMA_ALPHA = 0.15

# Sliding window size in frames (at 15 fps, 150 ≈ 10 seconds)
WINDOW_SIZE = 150

# Feature weights for final score
WEIGHTS = {
    "gaze":    0.30,
    "head":    0.25,
    "blink":   0.20,
    "posture": 0.15,
    "yawn":    0.10,
}


class FocusScorer:
    """
    Maintains a sliding window of feature dicts and produces a smoothed
    focus score each frame.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window: deque = deque(maxlen=window_size)
        self._blink_window: deque = deque(maxlen=window_size)
        self._prev_ear: float = 1.0
        self._blink_count: int = 0
        self._frames_per_second: float = 15.0  # updated by main loop
        self._ema_score: float = 0.85
        self._ema_active: bool = False
        
        # Adaptive EAR calibration for different eye shapes
        self.calibrated = False
        self.calibration_frames = 120  # ~8 seconds at 15 fps
        self.calibration_ears = []
        self.ear_baseline = 0.28
        self.ear_closed_thresh = 0.18
        
        # Drowsiness temporal counter (filters out blinks/jitter, triggers after ~1.5 seconds)
        self.drowsy_counter = 0
        self.drowsy_limit = 22

    def update(self, features: dict | None, fps: float = 15.0) -> dict:
        """
        Call once per frame.

        Args:
            features: dict from FeatureEngineer, or None if face not visible
            fps:      current capture FPS (used to normalise blink rate)

        Returns:
            {
              'score':  float [0.0 – 1.0],
              'state':  str  'FOCUSED' | 'DISTRACTED' | 'DROWSY' | 'AWAY',
              'components': dict of per-signal sub-scores,
            }
        """
        if features is None:
            # Face not visible
            score = self._smooth(0.0)
            return {"score": score, "state": "AWAY", "components": {}, "calibrated": self.calibrated}

        self._frames_per_second = max(fps, 1.0)
        self._window.append(features)

        ear = features["ear"]

        # ── Adaptive EAR Calibration ──────────────────────────────────────────
        if not self.calibrated:
            self.calibration_ears.append(ear)
            if len(self.calibration_ears) >= self.calibration_frames:
                # Use median to avoid blink outliers during calibration phase
                self.ear_baseline = float(np.median(self.calibration_ears))
                # Set closed threshold to 60% of baseline
                self.ear_closed_thresh = max(self.ear_baseline * 0.60, 0.12)
                self.calibrated = True
                print(f"[CALIBRATION] Complete. Baseline EAR: {self.ear_baseline:.3f}, Closed Thresh: {self.ear_closed_thresh:.3f}")

        # ── Blink detection (EAR crossing threshold) ─────────────────────────
        if self._prev_ear >= self.ear_closed_thresh > ear:
            self._blink_count += 1
        self._prev_ear = ear
        self._blink_window.append(self._blink_count)

        # ── Sub-score: Gaze (is the user looking at the screen?) ─────────────
        gaze_dev = abs(features["gaze_x"]) + abs(features["gaze_y"])
        gaze_score = max(0.0, 1.0 - gaze_dev / (GAZE_CENTRE_THRESH * 2))

        # ── Sub-score: Head Pose ──────────────────────────────────────────────
        yaw_ok   = 1.0 - min(abs(features["yaw"]),   HEAD_YAW_THRESH)   / HEAD_YAW_THRESH
        pitch_ok = 1.0 - min(abs(features["pitch"]), HEAD_PITCH_THRESH) / HEAD_PITCH_THRESH
        head_score = (yaw_ok * 0.6 + pitch_ok * 0.4)

        # ── Sub-score: Blink Rate ─────────────────────────────────────────────
        # Estimate blinks per minute over the current window
        window_secs = len(self._window) / self._frames_per_second
        blink_per_min = (self._blink_count / max(window_secs, 1)) * 60.0
        if BLINK_RATE_MIN <= blink_per_min <= BLINK_RATE_MAX:
            blink_score = 1.0
        else:
            dist = min(
                abs(blink_per_min - BLINK_RATE_MIN),
                abs(blink_per_min - BLINK_RATE_MAX)
            )
            blink_score = max(0.0, 1.0 - dist / 15.0)

        # ── Sub-score: Posture ────────────────────────────────────────────────
        pa = features["posture"]
        if pa <= POSTURE_GOOD_THRESH:
            posture_score = 1.0
        elif pa >= POSTURE_BAD_THRESH:
            posture_score = 0.0
        else:
            posture_score = 1.0 - (pa - POSTURE_GOOD_THRESH) / (POSTURE_BAD_THRESH - POSTURE_GOOD_THRESH)

        # ── Sub-score: No Yawn ────────────────────────────────────────────────
        yawn_score = 0.0 if features["mar"] > MAR_YAWN_THRESH else 1.0

        # ── Weighted aggregate ────────────────────────────────────────────────
        raw_score = (
            gaze_score    * WEIGHTS["gaze"] +
            head_score    * WEIGHTS["head"] +
            blink_score   * WEIGHTS["blink"] +
            posture_score * WEIGHTS["posture"] +
            yawn_score    * WEIGHTS["yawn"]
        )

        score = self._smooth(raw_score)

        # ── State classification ──────────────────────────────────────────────
        state = self._classify(score, features)

        return {
            "score": round(score, 3),
            "state": state,
            "components": {
                "gaze":    round(gaze_score,    3),
                "head":    round(head_score,    3),
                "blink":   round(blink_score,   3),
                "posture": round(posture_score, 3),
                "yawn":    round(yawn_score,    3),
            },
            "blink_per_min": round(blink_per_min, 1),
            "calibrated": self.calibrated,
        }

    def _smooth(self, value: float) -> float:
        if not self._ema_active:
            self._ema_score = value
            self._ema_active = True
        else:
            self._ema_score = EMA_ALPHA * value + (1 - EMA_ALPHA) * self._ema_score
        return round(float(self._ema_score), 3)

    def _classify(self, score: float, features: dict) -> str:
        ear = features.get("ear", 1.0)
        mar = features.get("mar", 0.0)
        yaw = abs(features.get("yaw", 0.0))

        # Check for immediate drowsiness indicators (small EAR or large MAR yawning)
        is_drowsy_signal = (ear < self.ear_closed_thresh) or (mar > MAR_YAWN_THRESH)

        if is_drowsy_signal:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 2)

        # Trigger drowsiness state only if indicators are sustained
        if self.drowsy_counter >= self.drowsy_limit:
            return "DROWSY"

        if yaw > HEAD_YAW_THRESH or score < 0.40:
            return "DISTRACTED"
        if score >= 0.70:
            return "FOCUSED"
        return "DISTRACTED"
