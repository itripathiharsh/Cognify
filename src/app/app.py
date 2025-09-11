# src/app/app.py
"""
Cognify / DriverEye - Full app.py (deployment-ready)

Key safe fixes included:
- st.session_state.user always initialized as a dict (Guest fallback)
- init_firebase() returns None if secrets are missing and won't call st.stop()
- asyncio exception handler to avoid aioice "call_exception_handler" NoneType errors on shutdown
- webrtc_streamer cleanup: Stop Camera button + ctx.stop() called safely
- atexit handler to attempt to stop running WebRTC context when process exits
All original app logic preserved where possible.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime
import pandas as pd
import joblib
from collections import deque
from fpdf import FPDF
import base64
import json
import atexit
import asyncio
import logging

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

# --- Add project root to PYTHONPATH ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Page Config ---
st.set_page_config(page_title="DriverEye / Cognify - Fatigue & Productivity", page_icon="🧠", layout="wide")

# -----------------------------
# Asyncio safe exception handler
# -----------------------------
# This avoids the "NoneType has no attribute 'call_exception_handler'" spam from aioice
def _safe_asyncio_exception_handler(loop, context):
    try:
        exc = context.get("exception")
        # If exception message shows the loop is closed / call_exception_handler missing, ignore it
        if exc is not None:
            msg = str(exc)
            if "call_exception_handler" in msg or "Transport is closed" in msg or "Socket is closed" in msg:
                # ignore noisy shutdown messages
                return
        # Fallback to default logging of other exceptions
        loop.default_exception_handler(context)
    except Exception:
        # Be defensive: don't crash exception handler itself
        logging.exception("Error in custom asyncio exception handler")

try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_safe_asyncio_exception_handler)
except Exception:
    # Some environments (rare) may not allow replacing handler — ignore in that case
    pass

# -----------------------------
# Ensure safe session_state defaults
# -----------------------------
if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = True

# Always keep user as a dict (avoid None or missing key crashes)
if "user" not in st.session_state or not isinstance(st.session_state.user, dict):
    st.session_state.user = {"uid": None, "email": "Guest", "display_name": "Guest"}

# db placeholder (set after init attempt)
if "db" not in st.session_state:
    st.session_state.db = None

# Session data buffers
if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=300)

if "history_buffer" not in st.session_state:
    st.session_state.history_buffer = pd.DataFrame(columns=['Time', 'Valence', 'Arousal', 'Fatigue Index', 'Productivity Score'])

# -----------------------------
# Firebase initialization (safe)
# -----------------------------
def init_firebase():
    """
    Initialize Firebase Admin SDK using Streamlit secrets.
    Returns Firestore client object, or None if not configured.
    Does not call st.stop() so app remains usable without Firebase.
    """
    try:
        # If already initialized, return client
        try:
            firebase_admin.get_app()
            return firestore.client()
        except ValueError:
            pass

        # Prefer Streamlit secrets for cloud deployment
        if 'firebase' in st.secrets:
            try:
                cred_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_dict)
                project_id = cred_dict.get("project_id", "emotion-productivity")
                bucket_name = f"{project_id}.appspot.com"
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
                st.info(f"✅ Firebase initialized with project: {project_id}")
                return firestore.client()
            except Exception as e:
                st.warning(f"⚠️ Firebase init failed: {e}")
                return None
        else:
            # If local debug and service account file exists, attempt that
            local_path = "firebase-service-account.json"
            if os.path.exists(local_path):
                try:
                    cred = credentials.Certificate(local_path)
                    firebase_admin.initialize_app(cred)
                    st.info("✅ Firebase initialized from local service account file.")
                    return firestore.client()
                except Exception as e:
                    st.warning(f"⚠️ Firebase local init failed: {e}")
                    return None

            st.info("ℹ️ Firebase not configured (no secrets or local key). Cloud features disabled.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Unexpected Firebase init error: {e}")
        return None

# Try to initialize Firebase once
if st.session_state.get("db") is None:
    st.session_state.db = init_firebase()

# -----------------------------
# Model loading helpers (kept same idea)
# -----------------------------
def load_fatigue_model_for_transformer():
    """Loads the FatigueCalculator model specifically for the VideoTransformer. Returns None on failure."""
    try:
        # Import locally to avoid import at module level if dependencies missing
        from src.fatigue.fatigue_calculator import FatigueCalculator
        fatigue_model = FatigueCalculator()
        print("✅ Fatigue model loaded for VideoTransformer")
        return fatigue_model
    except Exception as e:
        print(f"❌ Failed to load Fatigue model for VideoTransformer: {e}")
        return None

@st.cache_resource
def load_models_for_main_app():
    """Load any models needed in main app (e.g., personalized productivity model). Returns placeholders if missing."""
    try:
        prod_model = None
        personalized_model = None

        # Example local personalized model loading
        if st.session_state.get("user") and st.session_state.user.get("uid"):
            local_path = f"models/user_{st.session_state.user['uid']}_productivity_model.joblib"
            if os.path.exists(local_path):
                try:
                    personalized_model = joblib.load(local_path)
                    print("✅ Personalized model loaded for main app")
                except Exception as e:
                    print(f"⚠️ Could not load personalized model: {e}")

        return prod_model, personalized_model
    except Exception as e:
        st.warning(f"Failed to load main app models: {e}")
        return None, None

# Attempt to load main-app models (safe)
prod_model, personalized_model = load_models_for_main_app()

# -----------------------------
# WebRTC VideoTransformer
# -----------------------------
class FatigueVideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.fatigue_model = load_fatigue_model_for_transformer()
        if self.fatigue_model is None:
            self.disabled = True
            print("❌ VideoTransformer disabled due to missing fatigue model.")
        else:
            self.disabled = False
            self.data_log = deque(maxlen=400)  # store last ~20s at 20fps
            print("✅ FatigueVideoTransformer initialized.")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # If transformer disabled, return raw frame
        if self.disabled:
            return frame

        img_bgr = frame.to_ndarray(format="bgr24")
        try:
            # Update fatigue metrics (user-defined in your module)
            fatigue_data = self.fatigue_model.update_metrics(img_bgr)
            fatigue_index = self.fatigue_model.get_fatigue_index()

            # Simple productivity proxy (inverse of fatigue)
            display_productivity = max(1.0, 5.0 - (fatigue_index * 4))

            # Create data point (float cast to ensure JSONable/simple types)
            data_point = {
                'timestamp': float(time.time()),
                'fatigue_index': float(fatigue_index),
                'productivity_score': float(display_productivity),
                'ear': float(fatigue_data.get('ear', 0.0)),
                'mar': float(fatigue_data.get('mar', 0.0)),
                'valence': float(0.5 - fatigue_index),   # placeholder
                'arousal': float(0.5 + fatigue_index),   # placeholder
                'emotion_confidence': float(fatigue_data.get('emotion_confidence', 0.8))
            }
            # Append to local data log for UI reading
            self.data_log.append(data_point)

            # Overlay metrics onto frame for user
            color = (0, 255, 0)
            if fatigue_index >= 0.7:
                color = (0, 0, 255)
            elif fatigue_index >= 0.5:
                color = (0, 165, 255)

            cv2.putText(img_bgr, f"Fatigue Index: {fatigue_index:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img_bgr, f"Productivity: {display_productivity:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_bgr, f"EAR: {fatigue_data.get('ear', 0.0):.2f}  MAR: {fatigue_data.get('mar', 0.0):.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            # Don't crash transformer; overlay an error message.
            print(f"Error in transformer processing: {e}")
            cv2.putText(img_bgr, "Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -----------------------------
# Utility: try stopping running webrtc contexts on exit
# -----------------------------
# We'll store the last created webrtc context here so atexit can attempt to stop it cleanly.
_GLOBAL_WEBRTC_CTX = {"ctx": None}

def _atexit_cleanup():
    try:
        ctx = _GLOBAL_WEBRTC_CTX.get("ctx")
        if ctx is not None:
            try:
                # If playing, stop gracefully
                if hasattr(ctx, "stop"):
                    ctx.stop()
            except Exception:
                pass
    except Exception:
        pass

atexit.register(_atexit_cleanup)

# -----------------------------
# Feature engineering function (kept from original)
# -----------------------------
def engineer_realtime_features(data_buffer: deque, feature_names: list):
    """Engineers features for ML model from buffer."""
    if len(data_buffer) < 20:
        return None

    df = pd.DataFrame(list(data_buffer))
    features = {}

    metrics_to_agg = ['confidence', 'entropy', 'valence', 'arousal', 'EAR', 'MAR', 'FI']
    for col in metrics_to_agg:
        if col in df.columns:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_volatility'] = df[col].diff().std()

    if 'top1' in df.columns:
        emotion_histogram = df['top1'].value_counts(normalize=True).to_dict()
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            features[f'emotion_hist_{emotion}'] = emotion_histogram.get(emotion, 0.0)

    feature_df = pd.DataFrame([features])

    for col in feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    return feature_df[feature_names]

# -----------------------------
# Login page (kept from original)
# -----------------------------
def render_login_page():
    st.title("🔐 Login to DriverEye Dashboard")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state.user = {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': user.display_name or email.split('@')[0]
                }
                st.success(f"🎉 Welcome back, {st.session_state.user['display_name']}!")
                st.experimental_rerun()
            except Exception as e:
                st.error("❌ Login failed. (Admin SDK doesn't sign in like client SDK.)")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        name = st.text_input("Display Name", key="signup_name")
        if st.button("Sign Up", type="primary"):
            try:
                user = auth.create_user(email=email, password=password, display_name=name)
                st.session_state.user = {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': name or email.split('@')[0]
                }
                st.success("✅ Account created! Welcome!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Signup failed: {str(e)}")

# -----------------------------
# Dashboard page (kept original logic, added safe guards)
# -----------------------------
def render_dashboard_page():
    st.title("🧠 Real-time Productivity Dashboard")
    st.info("ℹ️ Webcam access is handled directly by your browser. Ensure you allow camera permissions. Baseline metrics are estimated automatically for WebRTC.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Camera Feed")

        # Create WebRTC streamer; assign to global so atexit can reference it
        webrtc_ctx = webrtc_streamer(
            key="driver-eye-fatigue-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FatigueVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        # Save for cleanup
        _GLOBAL_WEBRTC_CTX["ctx"] = webrtc_ctx

        # Show status
        if webrtc_ctx and webrtc_ctx.state.playing:
            st.success("✅ Webcam stream is active.")
            # Stop button to terminate gracefully
            if st.button("🛑 Stop Camera", key="stop_camera"):
                try:
                    webrtc_ctx.stop()
                    st.success("Stopped camera stream.")
                except Exception as e:
                    st.warning(f"Could not stop camera cleanly: {e}")
        else:
            st.info("⏸️ Webcam stream is inactive. Click 'Start' above (Play button on the video widget).")

        # Focus session controls
        st.markdown("### 🍅 Focus Session")
        col_pomo_start, col_pomo_status = st.columns(2)
        with col_pomo_start:
            if st.button("Start 25-min Focus Session", key="pomo_start"):
                st.session_state.focus_session_start = time.time()
                st.session_state.focus_session_active = True
                st.success("🍅 Focus session started!")
        with col_pomo_status:
            if st.session_state.get('focus_session_active', False):
                elapsed = time.time() - st.session_state.focus_session_start
                if elapsed > 25 * 60:
                    st.session_state.focus_session_active = False
                    st.session_state.focus_session_score = float(np.random.uniform(60, 95))
                    try:
                        st.toast(f"🎉 Session Complete! Estimated focus: {st.session_state.focus_session_score:.1f}%.", icon="🎉")
                    except Exception:
                        pass
                else:
                    remaining = max(0, 25 * 60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")

    with col2:
        st.header("📊 Real-time Metrics")
        # Placeholders
        fatigue_placeholder = st.empty()
        productivity_placeholder = st.empty()
        st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
        valence_placeholder = st.empty()
        arousal_placeholder = st.empty()
        st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
        ear_placeholder = st.empty()
        mar_placeholder = st.empty()
        st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
        confidence_placeholder = st.empty()

        st.header("📈 Live Trend Analysis")
        chart_placeholder = st.empty()

        # Gather data from transformer if active
        if _GLOBAL_WEBRTC_CTX.get("ctx") and getattr(_GLOBAL_WEBRTC_CTX["ctx"], "video_processor", None):
            video_processor = _GLOBAL_WEBRTC_CTX["ctx"].video_processor
            recent_data_points = list(getattr(video_processor, 'data_log', []))
            if recent_data_points:
                df_live = pd.DataFrame(recent_data_points)
                latest_point = df_live.iloc[-1]

                fatigue_index = float(latest_point.get('fatigue_index', 0.0))
                productivity_score = float(latest_point.get('productivity_score', 0.0))
                valence = float(latest_point.get('valence', 0.0))
                arousal = float(latest_point.get('arousal', 0.0))
                ear = float(latest_point.get('ear', 0.0))
                mar = float(latest_point.get('mar', 0.0))
                confidence = float(latest_point.get('emotion_confidence', 0.0))

                fatigue_color = "green" if fatigue_index < 0.3 else "orange" if fatigue_index < 0.6 else "red"
                fatigue_placeholder.markdown(
                    f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:{fatigue_color}; margin:0;'>{fatigue_index:.2f}</h2>",
                    unsafe_allow_html=True
                )
                productivity_placeholder.metric("📊 Productivity Score", f"{productivity_score:.2f}")
                valence_placeholder.metric("😊 Valence (Mood)", f"{valence:.2f}")
                arousal_placeholder.metric("⚡ Arousal (Energy)", f"{arousal:.2f}")
                ear_placeholder.metric("👁️ Eye Aspect Ratio", f"{ear:.3f}")
                mar_placeholder.metric("👄 Mouth Aspect Ratio", f"{mar:.3f}")
                confidence_placeholder.metric("🎯 Emotion Confidence", f"{confidence*100:.1f}%")

                # Chart
                try:
                    chart_df = df_live.tail(100).set_index('timestamp')
                    chart_placeholder.line_chart(chart_df[['fatigue_index', 'productivity_score']], color=["#FFA500", "#2E8B57"])
                except Exception:
                    chart_placeholder.line_chart(df_live[['fatigue_index', 'productivity_score']].tail(50))

                # Append to history buffer safely
                for _, row in df_live.iterrows():
                    new_row = pd.DataFrame([{
                        'Time': datetime.fromtimestamp(row['timestamp']),
                        'Valence': row.get('valence', 0.0),
                        'Arousal': row.get('arousal', 0.0),
                        'Fatigue Index': row.get('fatigue_index', 0.0),
                        'Productivity Score': row.get('productivity_score', 0.0)
                    }])
                    st.session_state.history_buffer = pd.concat([st.session_state.history_buffer, new_row], ignore_index=True)
                # Keep manageable size
                st.session_state.history_buffer = st.session_state.history_buffer.tail(1000)
            else:
                fatigue_placeholder.markdown("<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Waiting for data...</h2>", unsafe_allow_html=True)
                productivity_placeholder.metric("📊 Productivity Score", "N/A")
                chart_placeholder.info("Live chart will appear once the webcam stream is active.")
        else:
            fatigue_placeholder.markdown("<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Stream Off</h2>", unsafe_allow_html=True)
            productivity_placeholder.metric("📊 Productivity Score", "N/A")
            chart_placeholder.info("Start the video widget to see live metrics.")

# -----------------------------
# Session summary (preserve original functionality)
# -----------------------------
def render_session_summary_page():
    st.title("📊 Session Summary Report")

    if st.session_state.history_buffer.empty:
        st.warning("No session data yet. Complete a session first!")
        return

    df = st.session_state.history_buffer.copy()

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_prod = df['Productivity Score'].mean()
        st.metric("Average Productivity", f"{avg_prod:.2f}")
    with col2:
        peak_prod = df['Productivity Score'].max()
        st.metric("Peak Productivity", f"{peak_prod:.2f}")
    with col3:
        avg_fatigue = df['Fatigue Index'].mean()
        st.metric("Average Fatigue", f"{avg_fatigue:.2f}")

    # Charts
    st.header("📈 Productivity & Fatigue Over Time")
    try:
        st.line_chart(df.set_index('Time')[['Productivity Score', 'Fatigue Index']], color=["#2E8B57", "#FFA500"])
    except Exception:
        st.line_chart(df[['Productivity Score', 'Fatigue Index']].tail(200))

    # Peak/Low
    st.header("⏱️ Peak & Low Performance Periods")
    try:
        peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
        low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
        st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f})")
        st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f})")
    except Exception:
        st.write("Unable to compute peak/low periods for this session.")

    # Recommendations
    st.header("💡 Recommendations")
    if avg_fatigue > 0.5:
        st.warning("🛌 You showed high fatigue. Try shorter sessions with breaks.")
    if avg_prod < 3.0:
        st.info("🎯 Consider working during your peak hours (check Long-Term Trends page).")
    if peak_prod > 4.0:
        st.success("🚀 You hit high focus! Schedule important tasks during similar times.")

    # Export PDF
    st.header("📥 Export Report")
    if st.button("📄 Generate PDF Report"):
        if not df.empty:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 10, txt="DriverEye Session Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Average Productivity: {avg_prod:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Average Fatigue: {avg_fatigue:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Peak Productivity: {peak_prod:.2f}", ln=True)
            pdf_output = "session_report.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_output}">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No data available to generate report.")

    # Save to Firestore
    st.header("☁️ Save to Cloud")
    if st.button("💾 Save Session to Firebase"):
        if not st.session_state.get('user') or not st.session_state.user.get('uid'):
            st.warning("Please login first!")
            return
        if not st.session_state.db:
            st.warning("Firebase not configured. Configure secrets to enable cloud saving.")
            return
        try:
            db = st.session_state.db
            user_id = st.session_state.user['uid']
            df = st.session_state.history_buffer.copy()
            avg_prod = df['Productivity Score'].mean()
            peak_prod = df['Productivity Score'].max()
            avg_fatigue = df['Fatigue Index'].mean()
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'average_productivity': float(avg_prod),
                'peak_productivity': float(peak_prod),
                'average_fatigue': float(avg_fatigue),
                'duration_minutes': len(df) / 20 / 60 if len(df) > 0 else 0,
                'details': df.to_dict('records')
            }
            db.collection('users').document(user_id).collection('sessions').add(session_data)
            st.success("✅ Session saved to cloud!")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

# -----------------------------
# Long-term trends (kept original)
# -----------------------------
def render_long_term_trends_page():
    st.title("📈 Long-Term Trends & Insights")
    if not st.session_state.get('user') or not st.session_state.user.get('uid'):
        st.warning("Please login to see your trends!")
        return
    if not st.session_state.db:
        st.warning("Firebase not configured. Configure secrets to enable cloud trends.")
        return
    db = st.session_state.db
    user_id = st.session_state.user['uid']
    try:
        sessions_ref = db.collection('users').document(user_id).collection('sessions')
        docs = sessions_ref.stream()
        all_data = []
        for doc in docs:
            session = doc.to_dict()
            for detail in session.get('details', []):
                detail['SessionDate'] = session['timestamp'].split('T')[0]
                all_data.append(detail)
        if len(all_data) == 0:
            st.warning("No cloud data yet. Complete and save sessions first!")
            return
        df = pd.DataFrame(all_data)
        df['Time'] = pd.to_datetime(df['Time'])
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.day_name()
        df['Date'] = pd.to_datetime(df['SessionDate'])
        # Daily patterns
        st.header("⏰ Productivity by Hour of Day")
        hourly_avg = df.groupby('Hour')['Productivity Score'].mean()
        st.line_chart(hourly_avg)
        peak_hour = int(hourly_avg.idxmax())
        st.write(f"✅ **Your Peak Productivity Hour**: {peak_hour}:00")
        # Weekly patterns
        st.header("📅 Productivity by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily_avg = df.groupby('DayOfWeek')['Productivity Score'].mean().reindex(day_order)
        st.bar_chart(daily_avg)
        best_day = daily_avg.idxmax()
        worst_day = daily_avg.idxmin()
        st.write(f"🌟 **Best Day**: {best_day} | 📉 **Worst Day**: {worst_day}")
        # Fatigue trends
        st.header("📊 Fatigue Patterns")
        afternoon_df = df[df['Hour'] >= 16]
        morning_df = df[df['Hour'] < 12]
        if not afternoon_df.empty and not morning_df.empty:
            afternoon_fatigue = afternoon_df['Fatigue Index'].mean()
            morning_fatigue = morning_df['Fatigue Index'].mean()
            if morning_fatigue > 0 and afternoon_fatigue > morning_fatigue * 1.5:
                st.warning(f"⚠️ After 4 PM, your fatigue is {afternoon_fatigue/morning_fatigue:.1f}x higher than morning!")
        # Smart insights
        st.header("💡 Smart Insights")
        if peak_hour < 12:
            st.info("🌞 You're a morning person! Schedule deep work before noon.")
        else:
            st.info("🌙 You're an evening person! Save creative tasks for afternoon/evening.")
    except Exception as e:
        st.error(f"❌ Could not load trends: {e}")

# -----------------------------
# Main router / sidebar
# -----------------------------
st.sidebar.title("🧭 Navigation")

# If user not logged in, show login page
if not st.session_state.get("user") or not st.session_state.user.get("uid"):
    render_login_page()
else:
    st.sidebar.image("https://via.placeholder.com/50", width=50)
    st.sidebar.write(f"👤 **{st.session_state.user.get('display_name','Guest')}**")
    st.sidebar.write(f"📧 {st.session_state.user.get('email','')}")
    if st.sidebar.button("🚪 Logout"):
        # Reset safely
        st.session_state.user = {"uid": None, "email": "Guest", "display_name": "Guest"}
        st.experimental_rerun()

    page_selection = st.sidebar.radio(
        "Go to",
        ["Live Dashboard", "Session Summary", "Long-Term Trends"],
        label_visibility="collapsed"
    )

    if page_selection == "Live Dashboard":
        render_dashboard_page()
    elif page_selection == "Session Summary":
        render_session_summary_page()
    else:
        render_long_term_trends_page()
