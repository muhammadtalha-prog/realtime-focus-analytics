"""
src/main.py
-----------
Real-Time Focus & Cognitive Load Analytics — Webcam Loop
Runs the full landmark → feature → score pipeline at 20–30 FPS
and renders a premium OpenCV overlay.

Controls:
  Q or window close button → quit
"""

import sys
import os
import time

import cv2
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.landmark_extractor import LandmarkExtractor
from src.feature_engineer   import FeatureEngineer
from src.focus_scorer       import FocusScorer
from src.session_logger     import SessionLogger

WIN_TITLE = "Focus Analytics — Live"

# ─── Colour palette (BGR) ────────────────────────────────────────────────────
COL_FOCUSED    = (0,   220, 100)   # Green
COL_DISTRACTED = (30,  180, 255)   # Orange
COL_DROWSY     = (50,   50, 255)   # Red
COL_AWAY       = (120, 120, 120)   # Grey
COL_CYAN       = (255, 242,   0)   # Cyan accent
COL_WHITE      = (255, 255, 255)
COL_DIM        = (150, 150, 150)

STATE_COLOURS = {
    "FOCUSED":    COL_FOCUSED,
    "DISTRACTED": COL_DISTRACTED,
    "DROWSY":     COL_DROWSY,
    "AWAY":       COL_AWAY,
}


def state_colour(state: str):
    return STATE_COLOURS.get(state, COL_AWAY)


def draw_face_mesh(img, face_lms, h, w, colour=(0, 200, 200), alpha=0.55):
    """Draw a lightweight face mesh overlay (selected connections only)."""
    if face_lms is None:
        return

    # Key connection groups
    silhouette = list(range(0, 17))
    left_eye  = [33, 160, 158, 133, 153, 144, 33]
    right_eye = [362, 385, 387, 263, 373, 380, 362]
    lips_outer = [61, 291, 39, 181, 0, 17, 269, 405, 61]

    overlay = img.copy()

    def draw_poly(indices):
        pts = np.array(
            [[int(face_lms[i].x * w), int(face_lms[i].y * h)] for i in indices],
            dtype=np.int32
        )
        cv2.polylines(overlay, [pts], isClosed=False, color=colour, thickness=1, lineType=cv2.LINE_AA)

    for grp in [left_eye, right_eye, lips_outer]:
        draw_poly(grp)

    # Iris dots
    for idx in [474, 475, 476, 477, 469, 470, 471, 472]:
        try:
            x, y = int(face_lms[idx].x * w), int(face_lms[idx].y * h)
            cv2.circle(overlay, (x, y), 2, (255, 255, 100), -1, cv2.LINE_AA)
        except IndexError:
            pass

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_score_gauge(img, score: float, state: str, x: int, y: int, r: int = 70):
    """Draw a circular arc gauge for the focus score."""
    col = state_colour(state)

    # Background ring
    cv2.ellipse(img, (x, y), (r, r), -90, 0, 360, (50, 50, 50), 8, cv2.LINE_AA)
    # Score arc
    angle = int(360 * score)
    if angle > 0:
        cv2.ellipse(img, (x, y), (r, r), -90, 0, angle, col, 8, cv2.LINE_AA)

    # Score text
    score_str = f"{int(score * 100)}"
    ts = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    cv2.putText(img, score_str,
                (x - ts[0] // 2, y + ts[1] // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2, cv2.LINE_AA)
    cv2.putText(img, "FOCUS", (x - 20, y + r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_DIM, 1, cv2.LINE_AA)


def draw_metric_bar(img, label: str, value: float, x: int, y: int, bar_w: int = 120):
    """Compact horizontal mini-bar for a sub-score component."""
    bar_h = 7
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    filled = int(bar_w * max(0.0, min(1.0, value)))
    if filled > 0:
        hue = int(60 * value)
        bar_col = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0])
        cv2.rectangle(img, (x, y), (x + filled, y + bar_h), bar_col, -1)
    cv2.putText(img, label, (x, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_DIM, 1, cv2.LINE_AA)


def draw_hud(img, h: int, w: int, result: dict, features: dict | None,
             fps: float, elapsed_sec: float):
    """Render the right-side HUD panel."""
    panel_w = 230
    px0 = w - panel_w

    # Dark semi-transparent panel
    panel = img[:, px0:].copy()
    cv2.addWeighted(panel, 0.2, np.zeros_like(panel), 0.8, 0, dst=panel)
    img[:, px0:] = panel
    # Divider line
    cv2.line(img, (px0, 0), (px0, h), COL_CYAN[::-1], 1, cv2.LINE_AA)

    score  = result.get("score", 0.0)
    state  = result.get("state", "AWAY")
    comps  = result.get("components", {})
    bpm    = result.get("blink_per_min", 0.0)
    col    = state_colour(state)

    # ── Gauge ─────────────────────────────────────────────────────────────────
    gauge_cx = px0 + panel_w // 2
    draw_score_gauge(img, score, state, gauge_cx, 105, r=68)

    # ── State badge ───────────────────────────────────────────────────────────
    badge_x = px0 + (panel_w - 100) // 2
    cv2.rectangle(img, (badge_x, 185), (badge_x + 100, 208), col, -1)
    ts = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(img, state, (badge_x + (100 - ts[0]) // 2, 202),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

    py = 238
    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(img, "SIGNAL BREAKDOWN", (px0 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (px0 + 10, py + 4), (w - 10, py + 4), (60, 60, 60), 1)
    py += 20

    # ── Component bars ────────────────────────────────────────────────────────
    bar_labels = [
        ("GAZE",    comps.get("gaze", 0)),
        ("HEAD",    comps.get("head", 0)),
        ("BLINK",   comps.get("blink", 0)),
        ("POSTURE", comps.get("posture", 0)),
        ("YAWN",    comps.get("yawn", 0)),
    ]
    for lbl, val in bar_labels:
        draw_metric_bar(img, lbl, val, px0 + 10, py + 12, bar_w=panel_w - 22)
        py += 28

    py += 5
    cv2.line(img, (px0 + 10, py), (w - 10, py), (60, 60, 60), 1)
    py += 14

    # ── Raw feature readout ───────────────────────────────────────────────────
    if features:
        cv2.putText(img, "RAW FEATURES", (px0 + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_DIM, 1, cv2.LINE_AA)
        py += 16
        feat_lines = [
            f"EAR:    {features.get('ear',0):.3f}",
            f"MAR:    {features.get('mar',0):.3f}",
            f"GAZE:   ({features.get('gaze_x',0):+.2f}, {features.get('gaze_y',0):+.2f})",
            f"YAW:    {features.get('yaw',0):+.1f}°",
            f"PITCH:  {features.get('pitch',0):+.1f}°",
            f"POSTURE:{features.get('posture',0):.1f}°",
            f"BLINKS: {bpm:.0f}/min",
        ]
        for line in feat_lines:
            cv2.putText(img, line, (px0 + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COL_WHITE, 1, cv2.LINE_AA)
            py += 15

    # ── System stats ──────────────────────────────────────────────────────────
    py = h - 80
    cv2.line(img, (px0 + 10, py), (w - 10, py), (60, 60, 60), 1)
    py += 14
    mins, secs = divmod(int(elapsed_sec), 60)
    cv2.putText(img, f"SESSION  {mins:02d}:{secs:02d}", (px0 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_DIM, 1, cv2.LINE_AA)
    py += 15
    cv2.putText(img, f"FPS {fps:.0f}  |  PRESS Q TO QUIT", (px0 + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, COL_DIM, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    ok, frame = cap.read()
    if not ok:
        print("ERROR: Cannot open webcam.")
        return

    h, w = frame.shape[:2]
    print(f"Webcam: {w}×{h}")

    extractor = LandmarkExtractor()
    engineer  = FeatureEngineer()
    scorer    = FocusScorer()
    logger    = SessionLogger()
    logger.start_session()

    start_time = time.time()
    prev_time  = start_time
    fps        = 20.0

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Pipeline ──────────────────────────────────────────────────────────
        landmarks = extractor.extract(frame)
        features  = engineer.compute_features(landmarks, frame.shape)
        result    = scorer.update(features, fps)

        # ── Draw face mesh overlay ─────────────────────────────────────────────
        draw_face_mesh(frame, landmarks.get("face"), h, w,
                       colour=state_colour(result["state"]))

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, h, w, result, features, fps, now - start_time)

        # ── Logging ───────────────────────────────────────────────────────────
        logger.log(result, features)
        logger.write_shared_state(result, features, fps)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(WIN_TITLE, frame)

        # Clean exit: X button OR Q key
        try:
            if cv2.getWindowProperty(WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    logger.end_session()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
