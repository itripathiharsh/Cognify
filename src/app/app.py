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

# --- Always mark as calibrated for deployment ---
if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = True

st.title("🚗 DriverEye - Real-Time Fatigue Detection")

# --- FIREBASE INITIALIZATION ---
def init_firebase():
    """Initialize Firebase Admin SDK using Streamlit secrets"""
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            if 'firebase' in st.secrets:
                cred_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_dict)
                project_id = cred_dict.get("project_id", "emotion-productivity")
                bucket_name = f"{project_id}.appspot.com"
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
                print(f"✅ Firebase initialized with project: {project_id}")
                return firestore.client()
            else:
                st.error("❌ Firebase credentials not found in Streamlit secrets!")
                st.stop()
        except Exception as e:
            st.error(f"❌ Firebase init failed: {e}")
            st.stop()
    return firestore.client()

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
# --- Define the Video Processing Logic ---
class FatigueVideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.fatigue_model = load_fatigue_model_for_transformer()
        if self.fatigue_model is None:
            self.disabled = True
            print("❌ VideoTransformer disabled due to model loading failure.")
        else:
            self.disabled = False
            # Internal buffer to store recent processing results for data passing
            self.data_log = deque(maxlen=200) # Store last 200 data points (~10 secs at 20 fps)
            print("✅ Fatigue VideoTransformer initialized.")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.disabled:
            return frame

        img_bgr = frame.to_ndarray(format="bgr24")

        try:
            # --- PROCESSING LOGIC ---
            fatigue_data = self.fatigue_model.update_metrics(img_bgr)
            fatigue_index = self.fatigue_model.get_fatigue_index()

            # --- Simulate Productivity Score ---
            # In a full implementation, you would integrate emotion and scorer models here.
            # For demonstration, we'll derive a simple proxy from fatigue.
            # A more complex score could be: display_productivity = calculate_scores(emotion_data, fatigue_index)
            # Ensure it maps to your desired range, e.g., [1, 5]
            display_productivity = max(1.0, 5.0 - (fatigue_index * 4)) # Simple inverse relationship

            # --- Log data for passing back to main app ---
            # Store relevant metrics with a timestamp
            data_point = {
                'timestamp': time.time(),
                'fatigue_index': fatigue_index,
                'productivity_score': display_productivity,
                'ear': fatigue_data.get('ear', 0),
                'mar': fatigue_data.get('mar', 0),
                # Add valence, arousal, emotion confidence if available from full processing
                'valence': 0.5 - fatigue_index, # Placeholder
                'arousal': 0.5 + fatigue_index,  # Placeholder
                'emotion_confidence': 0.8 # Placeholder
            }
            self.data_log.append(data_point)

            # --- Overlay information on the frame ---
            color = (0, 255, 0) # Green
            if fatigue_index >= 0.7:
                color = (0, 0, 255) # Red
            elif fatigue_index >= 0.5:
                color = (0, 165, 255) # Orange

            cv2.putText(
                img_bgr,
                f"Fatigue Index: {fatigue_index:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            cv2.putText(
                img_bgr,
                f"Productivity: {display_productivity:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            # Optionally, display EAR and MAR values.
            cv2.putText(
                img_bgr,
                f"EAR: {fatigue_data['ear']:.2f}, MAR: {fatigue_data['mar']:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        except Exception as e:
            print(f"Error processing frame in transformer: {e}")
            cv2.putText(
                img_bgr,
                "Processing Error",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- MODEL & STATE INITIALIZATION (For main app logic) ---
@st.cache_resource
def load_models_for_main_app():
    """Placeholder for loading main app models if needed (e.g., personalized)."""
    try:
        # If you have models that run outside the transformer (e.g., post-session analysis)
        # from src.uncertainty.uncertainty_infer import UncertaintyInference
        # emotion_model = UncertaintyInference()
        prod_model = None
        personalized_model = None

        # Example loading a productivity model if it exists
        # try:
        #     prod_model = joblib.load("models/productivity_model.joblib")
        #     st.sidebar.success("✅ Productivity prediction model loaded")
        # except FileNotFoundError:
        #     prod_model = None
        #     st.sidebar.warning("⚠️ Productivity model not found.")

        if 'user' in st.session_state:
            local_path = f"models/user_{st.session_state.user['uid']}_productivity_model.joblib"
            if os.path.exists(local_path):
                try:
                    personalized_model = joblib.load(local_path)
                    print("✅ Personalized model loaded from local storage (main app)")
                except Exception as e:
                    print(f"⚠️ Could not load local personalized model: {e}")

        # return emotion_model, prod_model, personalized_model
        return None, prod_model, personalized_model # Return None for emotion_model if not used here
    except Exception as e:
        st.error(f"Failed to load models for main app: {e}")
        return None, None, None

# Initialize Firebase
if 'db' not in st.session_state:
    st.session_state.db = init_firebase()

# Load models for main app logic (if any)
# emotion_model_main, prod_model, personalized_model = load_models_for_main_app()
_, prod_model, personalized_model = load_models_for_main_app()

# Initialize session state for non-video parts
defaults = {
    'page': "Login",
    'is_calibrated': True, # Always calibrated for WebRTC
    'webcam_running': False, # Not directly used with webrtc_streamer anymore
    'data_buffer': deque(maxlen=300), # For storing session data points
    'history_buffer': pd.DataFrame(columns=['Time', 'Valence', 'Arousal', 'Fatigue Index', 'Productivity Score']), # For session/chart data
    'focus_session_active': False,
    'focus_session_start': 0,
    'focus_session_score': 0.0,
    'user': None # Will be set after login
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
                # Note: Firebase Admin SDK doesn't handle user passwords directly for login.
                # This is a simplified check. For real auth, use Firebase Client SDK or custom tokens.
                user = auth.get_user_by_email(email)
                st.session_state.user = {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': user.display_name or email.split('@')[0]
                }
                st.success(f"🎉 Welcome back, {st.session_state.user['display_name']}!")
                st.rerun()
            except Exception as e:
                st.error("❌ Login failed. Check email/password.")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password") # Not used by Admin SDK
        name = st.text_input("Display Name", key="signup_name")
        if st.button("Sign Up", type="primary"):
            try:
                # Requires enabling Email/Password provider in Firebase Console
                user = auth.create_user(
                    email=email,
                    password=password, # This might not work as expected with Admin SDK alone
                    display_name=name
                )
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
    st.info("ℹ️ Webcam access is handled directly by your browser. Ensure you allow camera permissions.")

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Camera Feed")

        # --- WEBCAM STREAM WITH WEBRTC ---
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
            st.info("⏸️ Webcam stream is inactive. Click 'Start' above.")

        # --- FOCUS SESSION CONTROLS ---
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
                    # Example: calculate a simple score based on average productivity during session
                    # This would ideally come from the data collected in st.session_state.data_buffer
                    st.session_state.focus_session_score = np.random.uniform(60, 95) # Dummy value
                    try:
                        st.toast(f"🎉 Session Complete! Estimated focus: {st.session_state.focus_session_score:.1f}%.", icon="🎉")
                    except:
                        pass # Ignore if Streamlit connection is lost
                else:
                    remaining = max(0, 25*60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")

    with col2:
        st.header("📊 Real-time Metrics")

        # --- Create placeholders for dynamic content ---
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

        # --- LIVE DATA FETCHING AND UPDATING ---
        # This block runs every time the script reruns (Streamlit's nature)
        # It checks the webrtc context for the latest data from the transformer
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            # Access the data log from the running VideoTransformer instance
            recent_data_points = list(getattr(webrtc_ctx.video_processor, 'data_log', []))

            if recent_data_points:
                # Convert to DataFrame for easier handling
                df_live = pd.DataFrame(recent_data_points)

                # --- Update Metrics ---
                # Get the latest data point
                latest_point = df_live.iloc[-1]
                fatigue_index = latest_point['fatigue_index']
                productivity_score = latest_point['productivity_score']
                valence = latest_point['valence']
                arousal = latest_point['arousal']
                ear = latest_point['ear']
                mar = latest_point['mar']
                confidence = latest_point['emotion_confidence']

                # Update placeholders with live data
                fatigue_color = "green" if fatigue_index < 0.3 else "orange" if fatigue_index < 0.6 else "red"
                fatigue_placeholder.markdown(
                    f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:{fatigue_color}; margin:0;'>{fatigue_index:.2f}</h2>",
                    unsafe_allow_html=True
                )
                productivity_placeholder.metric("📊 Productivity Score", f"{productivity_score:.2f}") # Adjust range if needed /5

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
                chart_placeholder.line_chart(
                    chart_df.set_index('timestamp')[['fatigue_index', 'productivity_score']],
                    color=["#FFA500", "#2E8B57"] # Orange for Fatigue, Green for Productivity
                )

                # --- Update Session State Buffers ---
                # Append live data to the main session buffer for session summary/history
                # Convert timestamp to datetime for consistency if needed
                for _, row in df_live.iterrows():
                     new_row = pd.DataFrame([{
                         'Time': datetime.fromtimestamp(row['timestamp']),
                         'Valence': row['valence'],
                         'Arousal': row['arousal'],
                         'Fatigue Index': row['fatigue_index'],
                         'Productivity Score': row['productivity_score']
                     }])
                     # Use pd.concat instead of append (append is deprecated)
                     st.session_state.history_buffer = pd.concat(
                         [st.session_state.history_buffer, new_row],
                         ignore_index=True
                     )
                # Keep buffer size manageable
                st.session_state.history_buffer = st.session_state.history_buffer.tail(1000) # Keep last 1000 points

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


# --- SESSION SUMMARY PAGE ---
def render_session_summary_page():
    st.title("📊 Session Summary Report")

    if st.session_state.history_buffer.empty:
        st.warning("No session data yet. Complete a session first!")
        return

    df = st.session_state.history_buffer.copy()
    # Ensure 'Time' column is datetime if it's not already (it should be from live updates)
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
        peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
        low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
        st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f})")
        st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f})")

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
        if 'user' not in st.session_state:
            st.warning("Please login first!")
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

# --- LONG-TERM TRENDS PAGE ---
def render_long_term_trends_page():
    st.title("📈 Long-Term Trends & Insights")

    if 'user' not in st.session_state:
        st.warning("Please login to see your trends!")
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
        df['Time'] = pd.to_datetime(df['Time']) # Ensure datetime
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
            if afternoon_fatigue > morning_fatigue * 1.5:
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

# --- MAIN ROUTER ---
st.sidebar.title("🧭 Navigation")

if 'user' not in st.session_state:
    render_login_page()
else:
    st.sidebar.image("https://via.placeholder.com/50", width=50)
    st.sidebar.write(f"👤 **{st.session_state.user['display_name']}**")
    if st.sidebar.button("🚪 Logout"):
        del st.session_state.user
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
