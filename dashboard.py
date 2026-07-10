"""
dashboard.py
------------
Streamlit live dashboard for the Focus Analytics System.

Run alongside src/main.py:
    Terminal 1: python src/main.py
    Terminal 2: streamlit run dashboard.py

Data source: shared_state.json (written by main.py every frame)
             + sessions.db   (SQLite for historical data)
"""

import json
import os
import time
import sqlite3
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Focus Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #0e1117; }

  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-card .label { font-size: 0.75rem; color: #888; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-card .value { font-size: 2.1rem; font-weight: 700; margin-top: 4px; }

  .state-focused    { color: #00dc64; }
  .state-distracted { color: #ffb347; }
  .state-drowsy     { color: #ff4b4b; }
  .state-away       { color: #888; }

  div[data-testid="stMetricValue"] > div { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

SHARED_STATE_PATH = "shared_state.json"
DB_PATH           = "sessions.db"
REFRESH_SEC       = 2   # Auto-refresh interval

# ─── Helper functions ─────────────────────────────────────────────────────────
def load_shared_state() -> dict:
    if not os.path.exists(SHARED_STATE_PATH):
        return {}
    try:
        with open(SHARED_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def state_css_class(state: str) -> str:
    return f"state-{state.lower()}"


def score_to_color(score: float) -> str:
    r = int(255 * (1 - score))
    g = int(220 * score)
    return f"rgb({r},{g},80)"


def load_current_session_history(session_id: str, limit: int = 900) -> pd.DataFrame:
    if not session_id or not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """SELECT ts, focus_score, state, ear, gaze_x, gaze_y, yaw, pitch, posture
               FROM measurements
               WHERE session_id = ?
               ORDER BY ts DESC LIMIT ?""",
            conn, params=(session_id, limit)
        )
        conn.close()
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_session_history_summary() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT s.started_at, s.duration_sec,
                   AVG(m.focus_score) as avg_score, COUNT(m.id) as measurements
            FROM sessions s
            LEFT JOIN measurements m ON s.session_id = m.session_id
            WHERE s.ended_at IS NOT NULL
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
            LIMIT 30
        """, conn)
        conn.close()
        df["started_at"] = pd.to_datetime(df["started_at"])
        df["duration_min"] = (df["duration_sec"] / 60).round(1)
        return df
    except Exception:
        return pd.DataFrame()


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_live, tab_history = st.tabs(["🟢  Live Dashboard", "📊  Session History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    state_data = load_shared_state()

    if not state_data:
        st.warning("⚠️  **main.py is not running.** Start it with `python src/main.py`, then refresh.")
        st.stop()

    age = time.time() - state_data.get("ts", 0)
    if age > 10:
        st.warning(f"⚠️  Data is {age:.0f}s old — is main.py still running?")

    score      = state_data.get("score", 0.0)
    state      = state_data.get("state", "AWAY")
    comps      = state_data.get("components", {})
    features   = state_data.get("features", {})
    fps_val    = state_data.get("fps", 0.0)
    session_id = state_data.get("session_id")
    bpm        = state_data.get("blink_per_min", 0.0)

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("## 🧠 Real-Time Focus Analytics")
        st.markdown(f"<span style='color:#888;font-size:0.8rem'>Session · {session_id[:8] if session_id else '—'}  &nbsp;|&nbsp;  Engine {fps_val:.0f} FPS</span>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='text-align:right;margin-top:12px'>"
                    f"<span class='{state_css_class(state)}' style='font-size:1.4rem;font-weight:700'>{state}</span>"
                    f"</div>", unsafe_allow_html=True)

    st.divider()

    # ── Score gauge + metrics ─────────────────────────────────────────────────
    g_col, m1, m2, m3, m4 = st.columns([2, 1, 1, 1, 1])

    with g_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 42, "color": score_to_color(score)}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
                "bar":  {"color": score_to_color(score), "thickness": 0.22},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,75,75,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,179,71,0.10)"},
                    {"range": [70,100], "color": "rgba(0,220,100,0.10)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": score * 100},
            },
            title={"text": "FOCUS SCORE", "font": {"size": 14, "color": "#888"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=240, margin=dict(l=20, r=20, t=40, b=10),
            font={"color": "white"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    metrics = [
        ("EAR", f"{features.get('ear', 0):.3f}", "Eye openness ratio"),
        ("GAZE X", f"{features.get('gaze_x', 0):+.2f}", "Left(-) / Right(+)"),
        ("YAW °", f"{features.get('yaw', 0):+.1f}°", "Head turn angle"),
        ("BLINKS/MIN", f"{bpm:.0f}", "Healthy: 10–20"),
    ]
    for col, (lbl, val, tooltip) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{lbl}</div>"
                f"<div class='value'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Component bars ────────────────────────────────────────────────────────
    st.markdown("#### Signal Breakdown")
    bar_cols = st.columns(5)
    comp_labels = ["GAZE", "HEAD", "BLINK", "POSTURE", "YAWN"]
    comp_keys   = ["gaze", "head", "blink", "posture", "yawn"]
    for col, lbl, key in zip(bar_cols, comp_labels, comp_keys):
        val = comps.get(key, 0.0)
        col.metric(lbl, f"{int(val * 100)}%")
        col.progress(val)

    st.divider()

    # ── Live time-series ──────────────────────────────────────────────────────
    st.markdown("#### Focus Score Over Session")
    df_live = load_current_session_history(session_id, limit=900)
    if not df_live.empty:
        fig_line = px.line(
            df_live, x="time", y="focus_score",
            color_discrete_sequence=["#00dc64"],
            labels={"focus_score": "Focus Score", "time": "Time"},
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(l=0, r=0, t=10, b=10),
            font={"color": "white"}, yaxis={"range": [0, 1]},
            xaxis={"gridcolor": "#1e2130"}, yaxis_gridcolor="#1e2130",
        )
        fig_line.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(0,220,100,0.05)", line_width=0)
        fig_line.add_hrect(y0=0.0, y1=0.4, fillcolor="rgba(255,75,75,0.05)", line_width=0)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Session data will appear here once measurements are logged.")

    # ── Break reminder ─────────────────────────────────────────────────────────
    if not df_live.empty and len(df_live) > 120:
        recent_avg = df_live["focus_score"].tail(120).mean()
        if recent_avg < 0.45:
            st.warning("🕐  **Your focus has been low for 2+ minutes.** Consider taking a short break!")

    # Auto-refresh
    time.sleep(REFRESH_SEC)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("## 📊 Session History")
    df_hist = load_session_history_summary()

    if df_hist.empty:
        st.info("No completed sessions yet. Run `python src/main.py` and press Q to end a session.")
    else:
        # ── Summary metrics ────────────────────────────────────────────────────
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Sessions",       len(df_hist))
        h2.metric("Avg Focus Score",      f"{df_hist['avg_score'].mean():.0%}")
        h3.metric("Total Focus Time",     f"{df_hist['duration_min'].sum():.0f} min")

        st.divider()

        # ── Score over sessions bar chart ──────────────────────────────────────
        st.markdown("#### Average Focus Score by Session")
        df_hist["label"] = df_hist["started_at"].dt.strftime("%b %d %H:%M")
        fig_bar = px.bar(
            df_hist.iloc[::-1], x="label", y="avg_score",
            color="avg_score", color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            labels={"avg_score": "Avg Focus", "label": "Session"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0, r=0, t=10, b=10),
            font={"color": "white"}, coloraxis_showscale=False,
            yaxis={"range": [0, 1], "gridcolor": "#1e2130"},
            xaxis={"gridcolor": "#1e2130"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Time-of-day focus heatmap ──────────────────────────────────────────
        st.markdown("#### Focus by Time of Day (last 7 days)")
        try:
            conn = sqlite3.connect(DB_PATH)
            df_tod = pd.read_sql_query("""
                SELECT ts, focus_score FROM measurements
                WHERE ts > strftime('%s', datetime('now', '-7 days'))
            """, conn)
            conn.close()
            if not df_tod.empty:
                df_tod["hour"] = pd.to_datetime(df_tod["ts"], unit="s").dt.hour
                df_tod["day"]  = pd.to_datetime(df_tod["ts"], unit="s").dt.day_name()
                pivot = df_tod.groupby(["day", "hour"])["focus_score"].mean().reset_index()
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                fig_heat = px.density_heatmap(
                    pivot, x="hour", y="day", z="focus_score",
                    color_continuous_scale="RdYlGn", range_color=[0, 1],
                    category_orders={"day": day_order},
                    labels={"focus_score": "Avg Focus", "hour": "Hour of Day", "day": "Day"},
                )
                fig_heat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(l=0, r=0, t=10, b=10),
                    font={"color": "white"},
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.caption("Heatmap requires 7 days of data.")

        # ── Raw session table ─────────────────────────────────────────────────
        st.markdown("#### Session Log")
        st.dataframe(
            df_hist[["started_at", "duration_min", "avg_score", "measurements"]]
            .rename(columns={
                "started_at": "Started",
                "duration_min": "Duration (min)",
                "avg_score": "Avg Focus",
                "measurements": "Data Points",
            }),
            use_container_width=True,
        )
