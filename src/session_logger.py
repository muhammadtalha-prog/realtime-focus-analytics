"""
session_logger.py
-----------------
Privacy-by-design SQLite logger.
  - NEVER stores raw video frames
  - Logs only derived numeric features + focus score
  - One measurement row per second (deduplicated by floor(timestamp))
"""

import sqlite3
import time
import uuid
import os
import json
from datetime import datetime, timedelta

DB_PATH = "sessions.db"


class SessionLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._session_id: str | None = None
        self._session_start: float | None = None
        self._last_log_ts: float = 0.0
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                started_at   TEXT,
                ended_at     TEXT,
                duration_sec REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                ts           REAL,
                focus_score  REAL,
                state        TEXT,
                ear          REAL,
                mar          REAL,
                gaze_x       REAL,
                gaze_y       REAL,
                yaw          REAL,
                pitch        REAL,
                roll         REAL,
                posture      REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()
        conn.close()

    def start_session(self):
        self._session_id    = str(uuid.uuid4())
        self._session_start = time.time()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
            (self._session_id, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        print(f"Session started: {self._session_id}")
        return self._session_id

    def log(self, focus_result: dict, features: dict | None):
        """Call every frame; internally rate-limits to once per second."""
        now = time.time()
        if now - self._last_log_ts < 1.0:
            return
        self._last_log_ts = now

        if not self._session_id:
            return

        f = features or {}
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO measurements
              (session_id, ts, focus_score, state, ear, mar,
               gaze_x, gaze_y, yaw, pitch, roll, posture)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            self._session_id,
            now,
            focus_result.get("score", 0.0),
            focus_result.get("state", "AWAY"),
            f.get("ear",     0.0),
            f.get("mar",     0.0),
            f.get("gaze_x",  0.0),
            f.get("gaze_y",  0.0),
            f.get("yaw",     0.0),
            f.get("pitch",   0.0),
            f.get("roll",    0.0),
            f.get("posture", 0.0),
        ))
        conn.commit()
        conn.close()

    def end_session(self):
        if not self._session_id:
            return
        duration = time.time() - (self._session_start or time.time())
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE sessions
            SET ended_at = ?, duration_sec = ?
            WHERE session_id = ?
        """, (datetime.utcnow().isoformat(), duration, self._session_id))
        conn.commit()
        conn.close()
        print(f"Session ended. Duration: {duration:.0f}s")
        self._session_id = None

    # ── Dashboard query helpers ───────────────────────────────────────────────
    def get_current_session_data(self, limit: int = 900) -> list[dict]:
        """Fetch the last `limit` measurements from the active session."""
        if not self._session_id:
            return []
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT ts, focus_score, state
            FROM measurements
            WHERE session_id = ?
            ORDER BY ts DESC LIMIT ?
        """, (self._session_id, limit)).fetchall()
        conn.close()
        return [{"ts": r[0], "score": r[1], "state": r[2]} for r in reversed(rows)]

    def get_session_history(self, n_days: int = 7) -> list[dict]:
        """Summarise sessions from the past n_days for the History tab."""
        cutoff = (datetime.utcnow() - timedelta(days=n_days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT s.started_at, s.duration_sec,
                   AVG(m.focus_score), COUNT(m.id)
            FROM sessions s
            LEFT JOIN measurements m ON s.session_id = m.session_id
            WHERE s.started_at > ?
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
        """, (cutoff,)).fetchall()
        conn.close()
        return [
            {
                "started_at": r[0],
                "duration_min": round((r[1] or 0) / 60, 1),
                "avg_score": round(r[2] or 0, 3),
                "measurements": r[3],
            }
            for r in rows
        ]

    def write_shared_state(self, focus_result: dict, features: dict | None,
                           fps: float, path: str = "shared_state.json"):
        """Write a tiny JSON file for dashboard polling (avoids network port)."""
        state_doc = {
            "ts":        time.time(),
            "score":     focus_result.get("score", 0.0),
            "state":     focus_result.get("state", "AWAY"),
            "components": focus_result.get("components", {}),
            "blink_per_min": focus_result.get("blink_per_min", 0.0),
            "features":  features or {},
            "fps":       round(fps, 1),
            "session_id": self._session_id,
        }
        try:
            with open(path, "w") as f:
                json.dump(state_doc, f)
        except Exception:
            pass
