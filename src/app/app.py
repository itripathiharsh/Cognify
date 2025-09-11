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
# import joblib # Imported conditionally if needed
from collections import deque
from fpdf import FPDF
import base64
import json
import atexit
import asyncio
import logging
import uuid

# --- Firebase Imports ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    st.warning("⚠️ Firebase libraries not available. Cloud features will be disabled.")

import asyncio
import logging

# -----------------------------
# Suppress noisy aioice errors after event loop closes
# -----------------------------
def ignore_aioice_errors(loop, context):
    msg = context.get("message", "")
    if "call_exception_handler" in msg or "sendto" in msg or "Transport is closed" in msg or "Socket is closed" in msg:
        # logging.debug(f"Ignored aioice error: {msg}") # Use debug to reduce log spam
        return
    loop.default_exception_handler(context)

# Apply the handler
try:
    loop = asyncio.get_event_loop()
    if loop and loop.is_running():
        # If loop is running, we might not be able to set the handler directly.
        # The custom handler inside VideoTransformer deals with errors during processing.
        pass
    else:
        loop.set_exception_handler(ignore_aioice_errors)
except Exception as e:
    # Fallback if getting loop fails
    try:
        asyncio.get_event_loop().set_exception_handler(ignore_aioice_errors)
    except Exception:
        pass

# -----------------------------
# Add project root to PYTHONPATH
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="DriverEye / Cognify - Fatigue & Productivity", page_icon="🧠", layout="wide")

# -----------------------------
# Asyncio safe exception handler (Fallback for older setups or if direct loop access fails)
# -----------------------------
def _safe_asyncio_exception_handler(loop, context):
    try:
        exc = context.get("exception")
        if exc is not None:
            msg = str(exc)
            # Common aioice/WebRTC shutdown messages to ignore
            if any(keyword in msg for keyword in [
                "call_exception_handler", "Transport is closed", "Socket is closed", "sendto"
            ]):
                return # Ignore these known, harmless shutdown errors
        # Pass other exceptions to the default handler
        loop.default_exception_handler(context)
    except Exception:
        # Defensive: don't let the error handler itself cause a crash
        logging.exception("Error in custom asyncio exception handler")

# Attempt to set the custom handler
try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_safe_asyncio_exception_handler)
except Exception:
    pass # Ignore if setting handler fails (e.g., in some Streamlit Cloud environments at boot)

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
    Returns Firestore client object, or None if not configured or libraries missing.
    Does not call st.stop() so app remains usable without Firebase.
    """
    if not FIREBASE_AVAILABLE:
        st.info("ℹ️ Firebase libraries not installed. Cloud features disabled.")
        return None

    try:
        # If already initialized, return client
        try:
            app = firebase_admin.get_app()
            return firestore.client()
        except ValueError:
            pass # Not initialized yet

        # Prefer Streamlit secrets for cloud deployment
        if 'firebase' in st.secrets:
            try:
                cred_dict = dict(st.secrets["firebase"])
                # cred = credentials.Certificate(cred_dict) # Passing dict directly often works
                project_id = cred_dict.get("project_id", "emotion-productivity")
                bucket_name = f"{project_id}.appspot.com"
                # Explicitly create cred if needed
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
                st.success(f"✅ Firebase initialized with project: {project_id}")
                return firestore.client()
            except Exception as e:
                st.warning(f"⚠️ Firebase init failed (secrets): {e}")
                return None
        else:
            # If local debug and service account file exists, attempt that
            local_path = "firebase-service-account.json" # Standard local filename
            if os.path.exists(local_path):
                try:
                    cred = credentials.Certificate(local_path)
                    firebase_admin.initialize_app(cred)
                    st.success("✅ Firebase initialized from local service account file.")
                    return firestore.client()
                except Exception as e:
                    st.warning(f"⚠️ Firebase local init failed: {e}")
                    return None

            st.info("ℹ️ Firebase not configured (no secrets or local key like 'firebase-service-account.json'). Cloud features disabled.")
            return None
    except Exception as e:
        st.error(f"❌ Unexpected Firebase init error: {e}")
        return None

# Try to initialize Firebase once
if st.session_state.get("db") is None and FIREBASE_AVAILABLE:
    st.session_state.db = init_firebase()
elif not FIREBASE_AVAILABLE:
     st.session_state.db = None # Explicitly set to None if lib missing

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
    # Conditional import for joblib if used
    try:
        import joblib
    except ImportError:
        st.warning("joblib not available for loading local models.")
        return None, None

    try:
        prod_model = None
        personalized_model = None

        # Example local personalized model loading
        if st.session_state.get("user") and st.session_state.user.get("uid"):
            local_path = f"models/user_{st.session_state.user['uid']}_productivity_model.joblib"
            if os.path.exists(local_path):
                try:
                    personalized_model = joblib.load(local_path)
                    st.sidebar.success("✅ Personalized model loaded.") # Inform user
                    print("✅ Personalized model loaded for main app")
                except Exception as e:
                    st.sidebar.warning("⚠️ Could not load personalized model.") # Inform user
                    print(f"⚠️ Could not load personalized model: {e}")

        # Load a general productivity model if it exists
        # prod_model_path = "models/productivity_model.joblib"
        # if os.path.exists(prod_model_path):
        #     try:
        #         prod_model = joblib.load(prod_model_path)
        #         print("✅ General productivity model loaded for main app")
        #     except Exception as e:
        #         print(f"⚠️ Could not load general productivity model: {e}")

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

        # Instance-specific error handler to catch issues during frame processing
        self._custom_exception_handler_set = False

    def _set_custom_handler_if_needed(self):
        if not self._custom_exception_handler_set:
            try:
                loop = asyncio.get_event_loop()
                original_handler = loop.get_exception_handler()
                if original_handler != _safe_asyncio_exception_handler:
                    loop.set_exception_handler(_safe_asyncio_exception_handler)
                self._custom_exception_handler_set = True
            except:
                pass # Silently fail if handler can't be set

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Safety check and set custom handler within transformer context if possible
        self._set_custom_handler_if_needed()

        # If transformer disabled, return raw frame
        if self.disabled:
            return frame

        img_bgr = frame.to_ndarray(format="bgr24")
        try:
            # Update fatigue metrics (user-defined in your module)
            fatigue_data = self.fatigue_model.update_metrics(img_bgr)
            fatigue_index = self.fatigue_model.get_fatigue_index()

            # Simple productivity proxy (inverse of fatigue) - Adjust logic as needed
            display_productivity = max(1.0, 5.0 - (fatigue_index * 4)) # Maps 0.0->5.0, 1.0->1.0

            # Create data point (ensure JSON serializable/simple types)
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
            color = (0, 255, 0) # Green
            if fatigue_index >= 0.7:
                color = (0, 0, 255) # Red
            elif fatigue_index >= 0.5:
                color = (0, 165, 255) # Orange

            cv2.putText(img_bgr, f"Fatigue Index: {fatigue_index:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img_bgr, f"Productivity: {display_productivity:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Optional: EAR and MAR
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
                    print("🧹 Cleaned up WebRTC context at exit.")
            except Exception as e:
                print(f"🧹 Warning during WebRTC cleanup at exit: {e}")
    except Exception as e:
        print(f"🧹 Error during atexit cleanup: {e}")

atexit.register(_atexit_cleanup)

# -----------------------------
# Feature engineering function (kept from original, simplified for demo)
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
# Login page (kept from original) but made robust to Admin SDK limitations
# -----------------------------
def render_login_page():
    st.title("🔐 Login to DriverEye Dashboard")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            # Admin SDK cannot verify client passwords directly in this server context.
            # This attempts lookup or gracefully falls back.
            try:
                if st.session_state.db and FIREBASE_AVAILABLE:
                    # If Firebase admin exists, try to fetch user by email
                    user = auth.get_user_by_email(email)
                    st.session_state.user = {
                        'uid': user.uid,
                        'email': user.email,
                        'display_name': user.display_name or email.split('@')[0]
                    }
                    st.success(f"🎉 Welcome back, {st.session_state.user['display_name']}! (Firebase user found)")
                    st.rerun() # Use st.rerun() for newer Streamlit versions
                else:
                    # No Firebase; create a local guest-style user
                    local_uid = "local_" + uuid.uuid4().hex[:8]
                    st.session_state.user = {
                        'uid': local_uid,
                        'email': email or f"guest_{local_uid}@local",
                        'display_name': email.split('@')[0] if email else f"Guest_{local_uid}"
                    }
                    st.success(f"🎉 Signed in locally as {st.session_state.user['display_name']}.")
                    st.rerun()
            except Exception as e:
                # Fallback: create local user session anyway (inform the user)
                local_uid = "local_" + uuid.uuid4().hex[:8]
                st.session_state.user = {
                    'uid': local_uid,
                    'email': email or f"guest_{local_uid}@local",
                    'display_name': email.split('@')[0] if email else f"Guest_{local_uid}"
                }
                st.warning(f"Logged in locally (Firebase lookup failed or unavailable): {e}")
                st.rerun()

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        name = st.text_input("Display Name", key="signup_name")
        if st.button("Sign Up", type="primary"):
            try:
                if st.session_state.db and FIREBASE_AVAILABLE:
                    # Try to create a Firebase user
                    user = auth.create_user(email=email, password=password, display_name=name)
                    st.session_state.user = {
                        'uid': user.uid,
                        'email': user.email,
                        'display_name': name or email.split('@')[0]
                    }
                    st.success("✅ Account created! Welcome!")
                    st.rerun()
                else:
                    # Create local user
                    local_uid = "local_" + uuid.uuid4().hex[:8]
                    st.session_state.user = {
                        'uid': local_uid,
                        'email': email or f"guest_{local_uid}@local",
                        'display_name': name or (email.split('@')[0] if email else f"Guest_{local_uid}")
                    }
                    st.success("✅ Local account created (no Firebase).")
                    st.rerun()
            except Exception as e:
                # Fallback local creation
                local_uid = "local_" + uuid.uuid4().hex[:8]
                st.session_state.user = {
                    'uid': local_uid,
                    'email': email or f"guest_{local_uid}@local",
                    'display_name': name or (email.split('@')[0] if email else f"Guest_{local_uid}")
                }
                st.warning(f"Created local account (Firebase create failed): {e}")
                st.rerun()

# -----------------------------
# Dashboard page (kept original logic, added safe guards)
# -----------------------------
def render_dashboard_page():
    st.title("🧠 Real-time Productivity Dashboard")
    st.info("ℹ️ Webcam access is handled directly by your browser. Ensure you allow camera permissions.")

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
        # Save context for potential cleanup or access later
        _GLOBAL_WEBRTC_CTX["ctx"] = webrtc_ctx

        # Show status and stop button
        if webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing:
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
                    # Example score calculation
                    st.session_state.focus_session_score = float(np.random.uniform(60, 95))
                    try:
                        st.toast(f"🎉 Session Complete! Estimated focus: {st.session_state.focus_session_score:.1f}%.", icon="🎉")
                    except Exception:
                        pass # Ignore if Streamlit version doesn't support toast
                else:
                    remaining = max(0, 25 * 60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")

    with col2:
        st.header("📊 Real-time Metrics")
        # Placeholders for dynamic content
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

        # --- LIVE DATA FETCHING AND UPDATING ---
        # This block runs every time the script reruns (Streamlit's nature)
        # It checks the webrtc context for the latest data from the transformer
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            # Access the data log from the running VideoTransformer instance
            recent_data_points = list(getattr(webrtc_ctx.video_processor, 'data_log', []))

            if recent_data_points:
                # Convert to DataFrame for easier handling
                df_live = pd.DataFrame(recent_data_points)
                # Ensure timestamp is numeric and convert to datetime for plotting if needed
                df_live['ts_dt'] = pd.to_datetime(df_live['timestamp'], unit='s', errors='coerce')

                # --- Update Metrics ---
                # Get the latest data point
                latest_point = df_live.iloc[-1]
                fatigue_index = float(latest_point.get('fatigue_index', 0.0))
                productivity_score = float(latest_point.get('productivity_score', 0.0))
                valence = float(latest_point.get('valence', 0.0))
                arousal = float(latest_point.get('arousal', 0.0))
                ear = float(latest_point.get('ear', 0.0))
                mar = float(latest_point.get('mar', 0.0))
                confidence = float(latest_point.get('emotion_confidence', 0.0))

                # Update UI placeholders with live data
                fatigue_color = "green" if fatigue_index < 0.3 else "orange" if fatigue_index < 0.6 else "red"
                fatigue_placeholder.markdown(
                    f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:{fatigue_color}; margin:0;'>{fatigue_index:.2f}</h2>",
                    unsafe_allow_html=True
                )
                productivity_placeholder.metric("📊 Productivity Score", f"{productivity_score:.2f}") # Adjust if out of 5

                valence_placeholder.metric("😊 Valence (Mood)", f"{valence:.2f}")
                arousal_placeholder.metric("⚡ Arousal (Energy)", f"{arousal:.2f}")

                ear_placeholder.metric("👁️ Eye Aspect Ratio", f"{ear:.3f}")
                mar_placeholder.metric("👄 Mouth Aspect Ratio", f"{mar:.3f}")
                confidence_placeholder.metric("🎯 Emotion Confidence", f"{confidence*100:.1f}%")

                # --- Update Chart ---
                # Display the last N points for a smoother live chart
                N = 100 # Show last 100 points
                chart_df = df_live.tail(N).copy()
                # Convert timestamp to readable time for plotting if needed, or use index
                try:
                    chart_placeholder.line_chart(
                        chart_df.set_index('ts_dt')[['fatigue_index', 'productivity_score']].dropna(),
                        color=["#FFA500", "#2E8B57"] # Orange for Fatigue, Green for Productivity
                    )
                except Exception as chart_e:
                    # Fallback if datetime conversion or charting fails
                    try:
                        chart_placeholder.line_chart(
                            chart_df[['fatigue_index', 'productivity_score']].tail(50).dropna()
                        )
                    except Exception:
                        chart_placeholder.info("Unable to draw live chart.")

                # --- Update Session State Buffers ---
                # Append live data to the main session buffer for session summary/history
                # Convert timestamp to datetime for consistency if needed
                appended_rows = []
                for _, row in df_live.iterrows():
                    try:
                        t = datetime.fromtimestamp(float(row['timestamp']))
                    except Exception:
                        t = pd.to_datetime(row.get('ts_dt', datetime.utcnow()))
                    new_row = pd.DataFrame([{
                        'Time': t,
                        'Valence': float(row.get('valence', 0.0)),
                        'Arousal': float(row.get('arousal', 0.0)),
                        'Fatigue Index': float(row.get('fatigue_index', 0.0)),
                        'Productivity Score': float(row.get('productivity_score', 0.0))
                    }])
                    appended_rows.append(new_row)

                if appended_rows:
                    # Use pd.concat for appending new dataframes
                    new_data_df = pd.concat(appended_rows, ignore_index=True)
                    # Concatenate with existing history
                    st.session_state.history_buffer = pd.concat(
                        [st.session_state.history_buffer, new_data_df],
                        ignore_index=True
                    )
                    # Keep buffer size manageable (e.g., last 1000 points)
                    st.session_state.history_buffer = st.session_state.history_buffer.tail(1000).reset_index(drop=True)

            else:
                # No data yet, show default/loading state
                fatigue_placeholder.markdown(
                    f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Loading...</h2>",
                    unsafe_allow_html=True
                )
                productivity_placeholder.metric("📊 Productivity Score", "N/A")
                # ... (set other placeholders to loading/default) ...
        else:
            # Stream not active, show static info or last known values
            fatigue_placeholder.markdown(
                f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Stream Off</h2>",
                unsafe_allow_html=True
            )
            productivity_placeholder.metric("📊 Productivity Score", "N/A")
            # ... (set other placeholders to off/default) ...
            chart_placeholder.info("Live chart will appear when the webcam stream is active.")


# -----------------------------
# Session summary (preserve original functionality)
# -----------------------------
def render_session_summary_page():
    st.title("📊 Session Summary Report")

    if st.session_state.history_buffer.empty:
        st.warning("No session data yet. Complete a session first!")
        return

    df = st.session_state.history_buffer.copy()
    # Ensure 'Time' column is datetime if it's not already
    # df['Time'] = pd.to_datetime(df['Time'])

    # --- Key Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_prod = df['Productivity Score'].mean()
        st.metric("Average Productivity", f"{avg_prod:.2f}") # Adjust units if out of 5
    with col2:
        peak_prod = df['Productivity Score'].max()
        st.metric("Peak Productivity", f"{peak_prod:.2f}")
    with col3:
        avg_fatigue = df['Fatigue Index'].mean()
        st.metric("Average Fatigue", f"{avg_fatigue:.2f}")

    # --- Charts ---
    st.header("📈 Productivity & Fatigue Over Time")
    if not df.empty:
        st.line_chart(df.set_index('Time')[['Productivity Score', 'Fatigue Index']],
                      color=["#2E8B57", "#FFA500"])
    else:
        st.write("No data to display.")

    # --- Peak/Low Periods ---
    st.header("⏱️ Peak & Low Performance Periods")
    if not df.empty:
        try:
            peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
            low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
            st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f})")
            st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f})")
        except Exception:
             st.write("Unable to compute peak/low periods for this session.")

    # --- Recommendations ---
    st.header("💡 Recommendations")
    if avg_fatigue > 0.5:
        st.warning("🛌 You showed high fatigue. Try shorter sessions with breaks.")
    if avg_prod < 3.0: # Adjust threshold based on your score range
        st.info("🎯 Consider working during your peak hours (check Long-Term Trends page).")
    if peak_prod > 4.0: # Adjust threshold
        st.success("🚀 You hit high focus! Schedule important tasks during similar times.")

    # --- Export PDF ---
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
            pdf.cell(200, 10, txt=f"Peak Productivity: {peak_prod:.2f} at {peak_time.strftime('%H:%M:%S')}", ln=True)

            # Save and offer download
            pdf_output = "session_report.pdf"
            pdf.output(pdf_output)

            with open(pdf_output, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_output}">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No data available to generate report.")

    # --- SAVE TO FIRESTORE ---
    st.header("☁️ Save to Cloud")
    if st.button("💾 Save Session to Firebase"):
        if not st.session_state.get('user') or not st.session_state.user.get('uid'):
            st.warning("Please login first!")
            return
        if not st.session_state.db:
             st.warning("Firebase not configured or unavailable. Cannot save to cloud.")
             return

        db = st.session_state.db
        user_id = st.session_state.user['uid']
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'average_productivity': float(avg_prod),
            'peak_productivity': float(peak_prod),
            'average_fatigue': float(avg_fatigue),
            'duration_minutes': len(df) / 20 / 60, # Approx based on 20fps
            'details': df.to_dict('records')
        }

        try:
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
        st.warning("Firebase not configured or unavailable. Cannot load cloud trends.")
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

        # --- Daily Patterns ---
        st.header("⏰ Productivity by Hour of Day")
        hourly_avg = df.groupby('Hour')['Productivity Score'].mean()
        st.line_chart(hourly_avg)

        peak_hour = hourly_avg.idxmax()
        st.write(f"✅ **Your Peak Productivity Hour**: {peak_hour}:00")

        # --- Weekly Patterns ---
        st.header("📅 Productivity by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = df.groupby('DayOfWeek')['Productivity Score'].mean().reindex(day_order)
        st.bar_chart(daily_avg)

        best_day = daily_avg.idxmax()
        worst_day = daily_avg.idxmin()
        st.write(f"🌟 **Best Day**: {best_day} | 📉 **Worst Day**: {worst_day}")

        # --- Fatigue Trends ---
        st.header("📊 Fatigue Patterns")
        # Example: Compare afternoon vs morning fatigue if data exists
        afternoon_df = df[df['Hour'] >= 16]
        morning_df = df[df['Hour'] < 12]
        if not afternoon_df.empty and not morning_df.empty:
            afternoon_fatigue = afternoon_df['Fatigue Index'].mean()
            morning_fatigue = morning_df['Fatigue Index'].mean()
            if morning_fatigue > 0 and afternoon_fatigue > morning_fatigue * 1.5:
                st.warning(f"⚠️ After 4 PM, your fatigue is {afternoon_fatigue/morning_fatigue:.1f}x higher than morning!")

        # --- Smart Insights ---
        st.header("💡 Smart Insights")
        if peak_hour < 12:
            st.info("🌞 You're a morning person! Schedule deep work before noon.")
        else:
            st.info("🌙 You're an evening person! Save creative tasks for afternoon/evening.")

        if worst_day in daily_avg.index and best_day in daily_avg.index:
            if daily_avg[worst_day] < daily_avg[best_day] * 0.7:
                st.warning(f"📅 Your productivity on {worst_day}s is {(1 - daily_avg[worst_day]/daily_avg[best_day])*100:.0f}% lower than {best_day}s. Consider lighter tasks those days.")

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
    # Sidebar user info
    st.sidebar.image("https://via.placeholder.com/50", width=50) # Replace with user avatar if available
    st.sidebar.write(f"👤 **{st.session_state.user.get('display_name', 'Guest')}**")
    # Optional: Display email if desired
    # st.sidebar.write(f"📧 {st.session_state.user.get('email', '')}")

    if st.sidebar.button("🚪 Logout"):
        # Reset safely to Guest state
        st.session_state.user = {"uid": None, "email": "Guest", "display_name": "Guest"}
        # Clear history on logout if desired
        # st.session_state.history_buffer = pd.DataFrame(columns=['Time', 'Valence', 'Arousal', 'Fatigue Index', 'Productivity Score'])
        st.rerun() # Use st.rerun() for newer Streamlit versions

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
