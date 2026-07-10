# 🧠 Real-Time Focus & Cognitive Load Analytics System

A privacy-first, webcam-based productivity tool that infers your attention and fatigue state in real time using MediaPipe landmark analysis — fully local, no raw video stored.

## What it does

| Signal | Feature | Meaning |
|---|---|---|
| Face Mesh | Eye Aspect Ratio (EAR) | Blink rate, drowsiness detection |
| Face Mesh (478-pt + iris) | Gaze vector | Where you're looking |
| Face Mesh (solvePnP) | Head pose yaw/pitch/roll | Head turning, nodding |
| Face Mesh | Mouth Aspect Ratio (MAR) | Yawn detection |
| Pose Landmarks | Posture slouch angle | Fatigue proxy via shoulder–ear geometry |

All 5 signals → **weighted sliding-window heuristic → Focus Score [0–100]** → EMA-smoothed for display

## Project structure

```
src/
  main.py              # Webcam loop + OpenCV HUD overlay
  landmark_extractor.py # MediaPipe FaceMesh + Pose wrapper
  feature_engineer.py  # EAR, gaze, head pose, posture math
  focus_scorer.py      # Heuristic sliding-window scorer + EMA
  session_logger.py    # SQLite logging + shared_state.json writer
  focus_model.py       # LSTM architecture stub (train on your own data)
dashboard.py           # Streamlit live dashboard + session history
```

## Quick start

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the webcam loop (Terminal 1)
python src/main.py

# 4. Run the dashboard (Terminal 2)
streamlit run dashboard.py
```

Models are auto-downloaded on first run (~15 MB total):
- `face_landmarker.task` (MediaPipe Face Mesh 478-pt)
- `pose_landmarker_lite.task` (MediaPipe Pose 33-pt)

## Architecture notes (CV/interview ready)

**Why landmark-based features instead of end-to-end CNN?**
- No labeled "focus" dataset exists for training
- Interpretable features = explainable AI (auditable per-signal scores)
- Runs at 25+ FPS on CPU — no GPU required
- Privacy by design: only numeric features stored, never raw video

**LSTM stub (`src/focus_model.py`)**
Train once you have collected ~1 hour of your own sessions:
```bash
python src/focus_model.py --train
```

## Controls
- **Q** or window **✕** → quit cleanly
- Dashboard auto-refreshes every 2 seconds
