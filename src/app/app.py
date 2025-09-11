import streamlit as st
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
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

#Firebase Imports
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Page Config
st.set_page_config(page_title="Real-time Productivity Dashboard", page_icon="🧠", layout="wide")

# FIREBASE INITIALIZATION

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


# MODEL LOADING (For use inside VideoTransformer)

def load_models_for_transformer():
    """Loads models specifically for the VideoTransformer."""
    try:
        from src.uncertainty.uncertainty_infer import UncertaintyInference
        from src.fatigue.fatigue_calculator import FatigueCalculator
        from src.product.scorer import calculate_scores

        emotion_model = UncertaintyInference()
        fatigue_model = FatigueCalculator()
        print("✅ Models loaded for VideoTransformer")
        return emotion_model, fatigue_model, calculate_scores
    except Exception as e:
        print(f"❌ Failed to load models for VideoTransformer: {e}")
        return None, None, None

#WEBCAM PROCESSING WITH WEBRTC

class EmotionFatigueVideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.emotion_model, self.fatigue_model, self.calculate_scores = load_models_for_transformer()
        if self.emotion_model is None or self.fatigue_model is None:
            self.disabled = True
            print("❌ VideoTransformer disabled due to model loading failure.")
        else:
            self.disabled = False
            print("✅ VideoTransformer initialized.")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.disabled:
            return frame

        img_bgr = frame.to_ndarray(format="bgr24")

        try:
            # PROCESSING LOGIC
            # 1. Fatigue Analysis
            fatigue_data = self.fatigue_model.update_metrics(img_bgr)
            fatigue_index = self.fatigue_model.get_fatigue_index()

            # 2. Emotion Analysis
            emotion_data = self.emotion_model.predict_with_uncertainty(img_bgr)

            # 3. Productivity Scoring (if emotion detected)
            display_productivity = "N/A"
            if emotion_data:
                final_scores = self.calculate_scores(emotion_data, fatigue_index)
                rule_prod = final_scores['productivity']
                display_productivity = round((rule_prod + 1) * 2 + 1, 2)
                display_productivity = np.clip(display_productivity, 1.0, 5.0)

            # Overlay information on the frame
            cv2.putText(img_bgr, f"Fatigue: {fatigue_index:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if fatigue_index < 0.5 else (0, 165, 255) if fatigue_index < 0.7 else (0, 0, 255), 2)
            cv2.putText(img_bgr, f"Productivity: {display_productivity}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Add emotion if detected
            if emotion_data:
                cv2.putText(img_bgr, f"Emotion: {emotion_data['top1']} ({emotion_data['confidence']:.2f})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error processing frame in transformer: {e}")
            cv2.putText(img_bgr, "Processing Error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# MODEL & STATE INITIALIZATION (For main app logic)

@st.cache_resource
def load_models_for_main_app():
    """Loads models for main app logic (e.g., personalized model if exists locally)."""
    try:
        from src.uncertainty.uncertainty_infer import UncertaintyInference
        from src.fatigue.fatigue_calculator import FatigueCalculator 
        from src.product.scorer import calculate_scores 

        emotion_model = UncertaintyInference()
        try:
            prod_model = joblib.load("models/productivity_model.joblib")
            st.sidebar.success("✅ Productivity prediction model loaded")
        except FileNotFoundError:
            prod_model = None
            st.sidebar.warning("⚠️ Productivity model not found. Using rule-based scoring.")

        voice_analyzer = None
        personalized_model = None

        if 'user' in st.session_state:
            local_path = f"models/user_{st.session_state.user['uid']}_productivity_model.joblib"
            if os.path.exists(local_path):
                try:
                    personalized_model = joblib.load(local_path)
                    print("✅ Personalized model loaded from local storage (main app)")
                except Exception as e:
                    print(f"⚠️ Could not load local personalized model: {e}")

        return emotion_model, prod_model, voice_analyzer, personalized_model
    except Exception as e:
        st.error(f"Failed to load models for main app: {e}")
        return None, None, None, None

# Initialize Firebase
if 'db' not in st.session_state:
    st.session_state.db = init_firebase()

# Load models for main app logic (personalized model, etc.)
emotion_model_main, prod_model, voice_analyzer, personalized_model = load_models_for_main_app()

if emotion_model_main is None:
    st.error("Application cannot start - main model loading failed.")
    st.stop()

# Initialize session state (mostly for non-webcam parts like focus session, login, history)
defaults = {
    'page': "Login",
    'is_calibrated': False, 
    'webcam_running': False, 
    'data_buffer': deque(maxlen=300),
    'history_buffer': pd.DataFrame(columns=['Time', 'Valence', 'Arousal', 'Fatigue Index', 'Productivity Score']),
    'last_pred_time': 0,
    'prod_model_features': prod_model.feature_names_in_ if prod_model else [],
    'fatigue_alert_shown': False,
    'focus_alert_shown': False,
    'focus_session_active': False,
    'focus_session_start': 0,
    'focus_data': [],
    'focus_session_score': 0.0,
    'voice_monitoring': False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# FEATURE ENGINEERING (For main app logic if needed)

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

# LOGIN PAGE

def render_login_page():
    st.title("🔐 Login to Productivity Dashboard")

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
                st.rerun()
            except Exception as e:
                st.error("❌ Login failed. Check email/password.")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        name = st.text_input("Display Name", key="signup_name")
        if st.button("Sign Up", type="primary"):
            try:
                user = auth.create_user(
                    email=email,
                    password=password,
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

#  DASHBOARD PAGES

def render_dashboard_page():
    st.title("🧠 Real-time Productivity Dashboard")

    # Calibration Note (WebRTC approach)
    st.info("ℹ️ Webcam access is handled directly by your browser. Ensure you allow camera permissions. Baseline metrics are established automatically.")

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Camera Feed")

        # WEBCAM STREAM WITH WEBRTC
        webrtc_ctx = webrtc_streamer(
            key="emotion-fatigue-analysis", 
            mode=WebRtcMode.SENDRECV,     
            video_processor_factory=EmotionFatigueVideoTransformer,
            media_stream_constraints={"video": True, "audio": False}, 
            async_processing=True,          
        )

     
        if webrtc_ctx.state.playing:
            st.write("✅ Webcam stream is active.")
        else:
            st.write("⏸️ Webcam stream is inactive. Click 'Start' above.")

        # FOCUS SESSION CONTROLS (keep existing logic)
        st.markdown("### 🍅 Focus Session")
        col_pomo_start, col_pomo_status = st.columns(2)
        with col_pomo_start:
            if st.button("Start 25-min Focus Session", key="pomo_start"):
                st.session_state.focus_session_start = time.time()
                st.session_state.focus_session_active = True
                st.session_state.focus_data = [] 
                st.success("🍅 Focus session started!")

        with col_pomo_status:
            if st.session_state.get('focus_session_active', False):
                elapsed = time.time() - st.session_state.focus_session_start
                if elapsed > 25*60:
                    st.session_state.focus_session_active = False
                    if len(st.session_state.focus_data) > 0:
                        st.session_state.focus_session_score = np.random.uniform(60, 95) # Dummy value
                        try:
                            st.toast(f"🎉 Session Complete! Estimated focus: {st.session_state.focus_session_score:.1f}%.", icon="🎉")
                        except:
                            pass
                    else:
                         # If no data, assume neutral or prompt user
                         st.session_state.focus_session_score = 50.0
                         st.info("ℹ️ Session ended. Data collection via WebRTC stream metrics is pending implementation.")
                else:
                    remaining = max(0, 25*60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")

    with col2:
        st.header("📊 Real-time Metrics")
        st.info("ℹ️ Metrics display via WebRTC stream is under development. Metrics below are from previous sessions or default.")

        confidence_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        ear_placeholder = st.empty()
        mar_placeholder = st.empty()
        fatigue_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        valence_placeholder = st.empty()
        arousal_placeholder = st.empty()
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        productivity_placeholder = st.empty()

        dummy_fatigue = 0.3
        dummy_productivity = 3.5
        dummy_valence = 0.2
        dummy_arousal = 0.4
        dummy_confidence = 0.85
        dummy_ear = 0.25
        dummy_mar = 0.35

        confidence_placeholder.metric("🎯 Emotion Confidence", f"{dummy_confidence*100:.1f}%", help="From last session")
        ear_placeholder.metric("👁️ Eye Aspect Ratio", f"{dummy_ear:.3f}")
        mar_placeholder.metric("👄 Mouth Aspect Ratio", f"{dummy_mar:.3f}")
        fatigue_color = "green" if dummy_fatigue < 0.3 else "orange" if dummy_fatigue < 0.6 else "red"
        fatigue_placeholder.markdown(
            f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:{fatigue_color}; margin:0;'>{dummy_fatigue:.2f}</h2>",
            unsafe_allow_html=True
        )
        valence_placeholder.metric("😊 Valence (Mood)", f"{dummy_valence:.2f}")
        arousal_placeholder.metric("⚡ Arousal (Energy)", f"{dummy_arousal:.2f}")
        productivity_placeholder.metric("📊 Productivity Score", f"{dummy_productivity}/5.0")
        
        st.header("📈 Trend Analysis (Last Session)")
        st.info("Live chart updates from WebRTC stream require advanced data passing.")
        if not st.session_state.history_buffer.empty:
             chart_df = st.session_state.history_buffer.set_index('Time')
             st.line_chart(chart_df[['Valence', 'Arousal', 'Fatigue Index', 'Productivity Score']],
                           color=["#FF4B4B", "#4B4BFF", "#FFA500", "#2E8B57"])
        else:
             st.write("No previous session data to display.")


#SESSION SUMMARY PAGE 
def render_session_summary_page():
    st.title("📊 Session Summary Report")

    if st.session_state.history_buffer.empty:
        st.warning("No session data yet. Complete a session first!")
        return

    df = st.session_state.history_buffer.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day_name()

    # Key Metrics 
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_prod = df['Productivity Score'].mean()
        st.metric("Average Productivity", f"{avg_prod:.2f}/5.0")
    with col2:
        peak_prod = df['Productivity Score'].max()
        st.metric("Peak Productivity", f"{peak_prod:.2f}/5.0")
    with col3:
        avg_fatigue = df['Fatigue Index'].mean()
        st.metric("Average Fatigue", f"{avg_fatigue:.2f}")

    #  Charts
    st.header("📈 Productivity & Fatigue Over Time")
    st.line_chart(df.set_index('Time')[['Productivity Score', 'Fatigue Index']],
                  color=["#2E8B57", "#FFA500"])

    # Peak/Low Periods
    st.header("⏱️ Peak & Low Performance Periods")
    peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
    low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
    st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f}/5.0)")
    st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f}/5.0)")

    #  Recommendations
    st.header("💡 Recommendations")
    if avg_fatigue > 0.5:
        st.warning("🛌 You showed high fatigue. Try shorter sessions with breaks.")
    if avg_prod < 3.0:
        st.info("🎯 Consider working during your peak hours (check Long-Term Trends page).")
    if peak_prod > 4.0:
        st.success("🚀 You hit high focus! Schedule important tasks during similar times.")

    #  Export PDF 
    st.header("📥 Export Report")
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Productivity Session Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Average Productivity: {avg_prod:.2f}/5.0", ln=True)
        pdf.cell(200, 10, txt=f"Average Fatigue: {avg_fatigue:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Peak Productivity: {peak_prod:.2f} at {peak_time.strftime('%H:%M:%S')}", ln=True)

        pdf_output = "session_report.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_output}">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    # SAVE TO FIRESTORE
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
            'duration_minutes': len(df) / 20 / 60,
            'details': df.to_dict('records')
        }

        try:
            db.collection('users').document(user_id).collection('sessions').add(session_data)
            st.success("✅ Session saved to cloud!")
        except Exception as e:
            st.error(f"❌ Save failed: {e}")


# LONG-TERM TRENDS PAGE 


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
        df['Time'] = pd.to_datetime(df['Time'])
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.day_name()
        df['Date'] = pd.to_datetime(df['SessionDate'])

        #Daily Patterns 
        st.header("⏰ Productivity by Hour of Day")
        hourly_avg = df.groupby('Hour')['Productivity Score'].mean()
        st.line_chart(hourly_avg)

        peak_hour = hourly_avg.idxmax()
        st.write(f"✅ **Your Peak Productivity Hour**: {peak_hour}:00")

        #  Weekly Patterns 
        st.header("📅 Productivity by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = df.groupby('DayOfWeek')['Productivity Score'].mean().reindex(day_order)
        st.bar_chart(daily_avg)

        best_day = daily_avg.idxmax()
        worst_day = daily_avg.idxmin()
        st.write(f"🌟 **Best Day**: {best_day} | 📉 **Worst Day**: {worst_day}")

        # Fatigue Trends
        st.header("📊 Fatigue Patterns")
        afternoon_fatigue = df[df['Hour'] >= 16]['Fatigue Index'].mean()
        morning_fatigue = df[df['Hour'] < 12]['Fatigue Index'].mean()
        if len(df[df['Hour'] >= 16]) > 0 and len(df[df['Hour'] < 12]) > 0:
            if afternoon_fatigue > morning_fatigue * 1.5:
                st.warning(f"⚠️ After 4 PM, your fatigue is {afternoon_fatigue/morning_fatigue:.1f}x higher than morning!")

        # Smart Insights 
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

# MAIN ROUTER 


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
