# src/app/app.py
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

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

# --- Add project root to PYTHONPATH ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Page Config ---
st.set_page_config(page_title="DriverEye - Fatigue Detection", page_icon="🚗", layout="wide")

# --- Ensure session_state keys exist and set safe defaults ---
if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = True

# Default user: keep as dict to avoid KeyErrors
if "user" not in st.session_state or not isinstance(st.session_state.user, dict):
    st.session_state.user = {"uid": None, "email": "Guest", "display_name": "Guest"}

# Default db placeholder (set after init attempt)
if "db" not in st.session_state:
    st.session_state.db = None

st.title("🚗 DriverEye - Real-Time Fatigue Detection")

# --- FIREBASE INITIALIZATION ---
def init_firebase():
    """
    Initialize Firebase Admin SDK using Streamlit secrets.
    Returns firestore client or None if not configured.
    """
    try:
        # If already initialized, just return client
        try:
            firebase_admin.get_app()
            return firestore.client()
        except ValueError:
            pass

        # Use Streamlit secrets if provided
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
            st.info("ℹ️ Firebase credentials not found in Streamlit secrets. Cloud features disabled.")
            return None
    except Exception as e:
        # Catch-all: don't crash the app for missing firebase
        st.warning(f"⚠️ Unexpected Firebase init error: {e}")
        return None

# Try initializing once (idempotent)
if st.session_state.get("db") is None:
    st.session_state.db = init_firebase()

# --- MODEL LOADING (For use inside VideoTransformer) ---
def load_fatigue_model_for_transformer():
    """Loads the FatigueCalculator model specifically for the VideoTransformer."""
    try:
        from src.fatigue.fatigue_calculator import FatigueCalculator
        fatigue_model = FatigueCalculator()
        print("✅ Fatigue model loaded for VideoTransformer")
        return fatigue_model
    except Exception as e:
        print(f"❌ Failed to load Fatigue model for VideoTransformer: {e}")
        return None

# --- WEBCAM PROCESSING WITH WEBRTC ---
class FatigueVideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.fatigue_model = load_fatigue_model_for_transformer()
        if self.fatigue_model is None:
            self.disabled = True
            print("❌ VideoTransformer disabled due to model loading failure.")
        else:
            self.disabled = False
            self.data_log = deque(maxlen=200)  # Store last N points for UI to read
            print("✅ Fatigue VideoTransformer initialized.")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.disabled:
            return frame

        img_bgr = frame.to_ndarray(format="bgr24")

        try:
            # Update fatigue metrics from frame
            fatigue_data = self.fatigue_model.update_metrics(img_bgr)
            fatigue_index = self.fatigue_model.get_fatigue_index()

            # Simple productivity proxy (inverse of fatigue)
            display_productivity = max(1.0, 5.0 - (fatigue_index * 4))

            # Prepare data point to be read by main app via webrtc_ctx.video_processor.data_log
            data_point = {
                'timestamp': time.time(),
                'fatigue_index': float(fatigue_index),
                'productivity_score': float(display_productivity),
                'ear': float(fatigue_data.get('ear', 0.0)),
                'mar': float(fatigue_data.get('mar', 0.0)),
                # placeholders for future multimodal inputs
                'valence': float(0.5 - fatigue_index),
                'arousal': float(0.5 + fatigue_index),
                'emotion_confidence': float(0.8)
            }
            self.data_log.append(data_point)

            # Overlay the metrics on the frame
            color = (0, 255, 0)
            if fatigue_index >= 0.7:
                color = (0, 0, 255)
            elif fatigue_index >= 0.5:
                color = (0, 165, 255)

            cv2.putText(img_bgr, f"Fatigue Index: {fatigue_index:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img_bgr, f"Productivity: {display_productivity:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_bgr,
                        f"EAR: {fatigue_data.get('ear', 0.0):.2f}  MAR: {fatigue_data.get('mar', 0.0):.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(f"Error processing frame in transformer: {e}")
            cv2.putText(img_bgr, "Processing Error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- MODEL & STATE INITIALIZATION (For main app logic) ---
@st.cache_resource
def load_models_for_main_app():
    """Placeholder for loading main app models if needed (e.g., personalized)."""
    try:
        prod_model = None
        personalized_model = None

        # Example: if user-specific model exists locally, try to load
        if st.session_state.get("user", {}).get("uid"):
            local_path = f"models/user_{st.session_state.user['uid']}_productivity_model.joblib"
            if os.path.exists(local_path):
                try:
                    personalized_model = joblib.load(local_path)
                    print("✅ Personalized model loaded for main app")
                except Exception as e:
                    print(f"⚠️ Could not load personalized model: {e}")

        return None, prod_model, personalized_model
    except Exception as e:
        st.error(f"Failed to load models for main app: {e}")
        return None, None, None

# Attempt to load models (safe if missing)
_, prod_model, personalized_model = load_models_for_main_app()

# --- Initialize session state defaults (preserve existing structures) ---
defaults = {
    'page': "Login",
    'is_calibrated': True,  # WebRTC approach
    'webcam_running': False,
    'data_buffer': deque(maxlen=300),
    'history_buffer': pd.DataFrame(columns=['Time', 'Valence', 'Arousal', 'Fatigue Index', 'Productivity Score']),
    'focus_session_active': False,
    'focus_session_start': 0,
    'focus_session_score': 0.0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- LOGIN PAGE ---
def render_login_page():
    st.title("🔐 Login to DriverEye Dashboard")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            try:
                # NOTE: Admin SDK can't perform client-style sign-in with password.
                # This is a simplified lookup. For production, use proper client auth flows.
                user = auth.get_user_by_email(email)
                st.session_state.user = {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': user.display_name or email.split('@')[0]
                }
                st.success(f"🎉 Welcome back, {st.session_state.user['display_name']}!")
                st.rerun()
            except Exception as e:
                st.error("❌ Login failed. Check email/password or use a proper client-auth flow.")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        name = st.text_input("Display Name", key="signup_name")
        if st.button("Sign Up", type="primary"):
            try:
                # Warning: Admin SDK create_user may work but doesn't implement client-side password auth.
                user = auth.create_user(email=email, password=password, display_name=name)
                st.session_state.user = {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': name or email.split('@')[0]
                }
                st.success("✅ Account created! Welcome!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Signup failed: {str(e)}")

# --- DASHBOARD PAGE ---
def render_dashboard_page():
    st.info("ℹ️ Webcam access is handled directly by your browser. Allow camera permission when prompted. Baseline metrics are automatically estimated.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Camera Feed")

        webrtc_ctx = webrtc_streamer(
            key="driver-eye-fatigue-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FatigueVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.success("✅ Webcam stream is active.")
        else:
            st.info("⏸️ Webcam stream is inactive. Click 'Start' above in the video widget.")

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
                if elapsed > 25*60:
                    st.session_state.focus_session_active = False
                    # Placeholder: compute a focus score (would come from collected metrics)
                    st.session_state.focus_session_score = np.random.uniform(60, 95)
                    try:
                        st.toast(f"🎉 Session Complete! Estimated focus: {st.session_state.focus_session_score:.1f}%.", icon="🎉")
                    except:
                        pass
                else:
                    remaining = max(0, 25*60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")

    with col2:
        st.header("📊 Real-time Metrics")
        # Placeholders
        fatigue_placeholder = st.empty()
        productivity_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        valence_placeholder = st.empty()
        arousal_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        ear_placeholder = st.empty()
        mar_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        confidence_placeholder = st.empty()

        st.header("📈 Live Trend Analysis")
        chart_placeholder = st.empty()

        # Pull recent data from the running transformer instance if available
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            recent_data_points = list(getattr(webrtc_ctx.video_processor, 'data_log', []))
            if recent_data_points:
                df_live = pd.DataFrame(recent_data_points)
                latest_point = df_live.iloc[-1]

                fatigue_index = float(latest_point['fatigue_index'])
                productivity_score = float(latest_point['productivity_score'])
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

                # Chart last N points
                N = 100
                chart_df = df_live.tail(N).copy()
                # The transformer timestamps are epoch seconds; use them as index for quick plotting
                try:
                    chart_df_indexed = chart_df.set_index('timestamp')
                    chart_placeholder.line_chart(
                        chart_df_indexed[['fatigue_index', 'productivity_score']],
                        color=["#FFA500", "#2E8B57"]
                    )
                except Exception:
                    chart_placeholder.line_chart(
                        chart_df[['fatigue_index', 'productivity_score']].tail(50),
                        color=["#FFA500", "#2E8B57"]
                    )

                # Append latest points to session history buffer
                # Convert timestamps and append safely
                for _, row in df_live.iterrows():
                    new_row = pd.DataFrame([{
                        'Time': datetime.fromtimestamp(row['timestamp']),
                        'Valence': row.get('valence', 0.0),
                        'Arousal': row.get('arousal', 0.0),
                        'Fatigue Index': row.get('fatigue_index', 0.0),
                        'Productivity Score': row.get('productivity_score', 0.0)
                    }])
                    st.session_state.history_buffer = pd.concat(
                        [st.session_state.history_buffer, new_row],
                        ignore_index=True
                    )
                # Keep history manageable
                st.session_state.history_buffer = st.session_state.history_buffer.tail(1000)
            else:
                fatigue_placeholder.markdown(
                    "<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Loading...</h2>",
                    unsafe_allow_html=True
                )
                productivity_placeholder.metric("📊 Productivity Score", "N/A")
        else:
            fatigue_placeholder.markdown(
                "<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:gray; margin:0;'>Stream Off</h2>",
                unsafe_allow_html=True
            )
            productivity_placeholder.metric("📊 Productivity Score", "N/A")
            chart_placeholder.info("Live chart will appear when the webcam stream is active.")

# --- SESSION SUMMARY PAGE ---
def render_session_summary_page():
    st.title("📊 Session Summary Report")

    if st.session_state.history_buffer.empty:
        st.warning("No session data yet. Complete a session first!")
        return

    df = st.session_state.history_buffer.copy()

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_prod = df['Productivity Score'].mean() if not df.empty else 0.0
        st.metric("Average Productivity", f"{avg_prod:.2f}")
    with col2:
        peak_prod = df['Productivity Score'].max() if not df.empty else 0.0
        st.metric("Peak Productivity", f"{peak_prod:.2f}")
    with col3:
        avg_fatigue = df['Fatigue Index'].mean() if not df.empty else 0.0
        st.metric("Average Fatigue", f"{avg_fatigue:.2f}")

    # Charts
    st.header("📈 Productivity & Fatigue Over Time")
    if not df.empty:
        try:
            st.line_chart(df.set_index('Time')[['Productivity Score', 'Fatigue Index']],
                          color=["#2E8B57", "#FFA500"])
        except Exception:
            st.write("Unable to plot time-indexed chart; showing basic chart.")
            st.line_chart(df[['Productivity Score', 'Fatigue Index']].tail(200))
    else:
        st.write("No data to display.")

    # Peak/Low periods
    st.header("⏱️ Peak & Low Performance Periods")
    if not df.empty:
        try:
            peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
            low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
            st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f})")
            st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f})")
        except Exception:
            st.write("Unable to compute peak/low times.")

    # Recommendations
    st.header("💡 Recommendations")
    if 'avg_fatigue' in locals() and avg_fatigue > 0.5:
        st.warning("🛌 You showed high fatigue. Try shorter sessions with breaks.")
    if 'avg_prod' in locals() and avg_prod < 3.0:
        st.info("🎯 Consider working during your peak hours (check Long-Term Trends page).")
    if 'peak_prod' in locals() and peak_prod > 4.0:
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
        try:
            db.collection('users').document(user_id).collection('sessions').add(session_data)
            st.success("✅ Session saved to cloud!")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")

# --- LONG-TERM TRENDS PAGE ---
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

# --- MAIN ROUTER / SIDEBAR ---
st.sidebar.title("🧭 Navigation")

# Show login page if user not logged in (uid missing)
if not st.session_state.get("user") or not st.session_state.user.get("uid"):
    render_login_page()
else:
    st.sidebar.image("https://via.placeholder.com/50", width=50)
    st.sidebar.write(f"👤 **{st.session_state.user.get('display_name','Guest')}**")
    st.sidebar.write(f"📧 {st.session_state.user.get('email','')}")
    if st.sidebar.button("🚪 Logout"):
        # Reset to safe guest values
        st.session_state.user = {"uid": None, "email": "Guest", "display_name": "Guest"}
        st.rerun()

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
