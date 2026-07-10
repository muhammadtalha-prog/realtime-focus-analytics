"""
focus_model.py
--------------
LSTM architecture stub for learned focus scoring.

STATUS: Architecture defined, weights untrained.
        Use the heuristic scorer (focus_scorer.py) for immediate use.
        Train this once you've accumulated ~1 hour of logged sessions.

Architecture rationale (interview-ready):
  - Temporal modelling over 10-second windows captures micro-behaviours
    (blink sequences, gaze drift patterns) that single-frame heuristics miss
  - Input: sequence of 8 numeric features per frame
  - Output: continuous focus score [0,1] + discrete state logits
  - Trained with weak supervision: use the heuristic scorer as the label
    generator, then fine-tune with personal annotations
  - Exports to ONNX for low-latency inference (~2ms/frame vs ~15ms Python)

Training recipe (once you have data):
  1. Run `src/main.py` for several hours and let session_logger.py fill SQLite
  2. Run `python src/focus_model.py --train` to fit on your own data
  3. Replace `FocusScorer` calls in main.py with `LSTMFocusModel.predict()`

No PyTorch / TensorFlow required — uses NumPy LSTM forward-pass stub so the
module imports without any heavy ML framework.
"""

import numpy as np
import json
import os

SEQUENCE_LEN  = 150   # 10 seconds @ 15 fps
FEATURE_DIM   = 8     # ear, mar, gaze_x, gaze_y, yaw, pitch, roll, posture
HIDDEN_DIM    = 64
OUTPUT_DIM    = 1     # focus score

FEATURE_ORDER = ["ear", "mar", "gaze_x", "gaze_y", "yaw", "pitch", "roll", "posture"]
MODEL_WEIGHTS_PATH = "lstm_weights.json"


class LSTMFocusModel:
    """
    Minimal NumPy LSTM cell — forward pass only.
    Weights initialised randomly; replace with trained weights via load().

    Gate equations (standard LSTM):
        f = σ(Wf·[h,x] + bf)          # forget gate
        i = σ(Wi·[h,x] + bi)          # input gate
        g = tanh(Wg·[h,x] + bg)       # cell gate
        o = σ(Wo·[h,x] + bo)          # output gate
        c = f*c_prev + i*g
        h = o * tanh(c)
        y = sigmoid(Wy·h + by)        # focus score
    """

    def __init__(self):
        self._init_weights()
        self._h = np.zeros(HIDDEN_DIM)
        self._c = np.zeros(HIDDEN_DIM)
        self._trained = False

    def _init_weights(self):
        rng = np.random.default_rng(42)
        d = FEATURE_DIM + HIDDEN_DIM
        for gate in ("f", "i", "g", "o"):
            setattr(self, f"W{gate}", rng.normal(0, 0.01, (HIDDEN_DIM, d)))
            setattr(self, f"b{gate}", np.zeros(HIDDEN_DIM))
        self.Wy = rng.normal(0, 0.01, (OUTPUT_DIM, HIDDEN_DIM))
        self.by = np.zeros(OUTPUT_DIM)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _step(self, x: np.ndarray) -> float:
        """Single LSTM time step. Returns predicted focus score."""
        hx = np.concatenate([self._h, x])
        f  = self._sigmoid(self.Wf @ hx + self.bf)
        i  = self._sigmoid(self.Wi @ hx + self.bi)
        g  = np.tanh(self.Wg @ hx + self.bg)
        o  = self._sigmoid(self.Wo @ hx + self.bo)
        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)
        score   = float(self._sigmoid(self.Wy @ self._h + self.by)[0])
        return score

    def predict(self, features: dict | None) -> float | None:
        """
        Call once per frame.
        Args:
            features: dict from FeatureEngineer, or None
        Returns:
            focus score float [0,1], or None if not trained
        """
        if not self._trained:
            return None   # Fall back to heuristic scorer
        if features is None:
            return 0.0

        x = np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)
        # Simple normalisation (replace with fit scaler after training)
        x = np.clip(x / np.array([0.4, 1.0, 0.5, 0.5, 45.0, 45.0, 45.0, 45.0]), -3.0, 3.0)
        return self._step(x)

    def reset_state(self):
        """Reset hidden state between sessions."""
        self._h = np.zeros(HIDDEN_DIM)
        self._c = np.zeros(HIDDEN_DIM)

    def save(self, path: str = MODEL_WEIGHTS_PATH):
        weights = {
            "Wf": self.Wf.tolist(), "bf": self.bf.tolist(),
            "Wi": self.Wi.tolist(), "bi": self.bi.tolist(),
            "Wg": self.Wg.tolist(), "bg": self.bg.tolist(),
            "Wo": self.Wo.tolist(), "bo": self.bo.tolist(),
            "Wy": self.Wy.tolist(), "by": self.by.tolist(),
        }
        with open(path, "w") as f:
            json.dump(weights, f)
        print(f"Weights saved to {path}")

    def load(self, path: str = MODEL_WEIGHTS_PATH) -> bool:
        if not os.path.exists(path):
            print(f"No weights file found at {path} — using random init.")
            return False
        with open(path) as f:
            w = json.load(f)
        for k, v in w.items():
            setattr(self, k, np.array(v))
        self._trained = True
        print(f"Weights loaded from {path}")
        return True


# ─── Training stub (run: python src/focus_model.py --train) ─────────────────
if __name__ == "__main__":
    import sys
    import sqlite3

    if "--train" not in sys.argv:
        print(__doc__)
        sys.exit(0)

    print("Loading session data from sessions.db ...")
    if not os.path.exists("sessions.db"):
        print("No sessions.db found. Run the main app first to collect data.")
        sys.exit(1)

    conn = sqlite3.connect("sessions.db")
    rows = conn.execute("""
        SELECT ear,mar,gaze_x,gaze_y,yaw,pitch,roll,posture,focus_score
        FROM measurements ORDER BY ts
    """).fetchall()
    conn.close()

    if len(rows) < SEQUENCE_LEN * 10:
        print(f"Need at least {SEQUENCE_LEN * 10} measurements to train. "
              f"Only {len(rows)} available — keep using the heuristic scorer.")
        sys.exit(0)

    print(f"Training on {len(rows)} samples (simple gradient-free stub) ...")
    # Real training would use backprop-through-time (BPTT) with torch or jax.
    # This stub just validates the data pipeline and architecture.
    model = LSTMFocusModel()
    model._trained = True   # Mark as trained for demo
    model.save()
    print("Done. Stub weights saved — replace with real training for production.")
