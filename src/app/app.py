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

# Firebase Imports 
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

# Add project root to PYTHONPATH 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Modules
from src.uncertainty.uncertainty_infer import UncertaintyInference
from src.fatigue.fatigue_calculator import FatigueCalculator
from src.product.scorer import calculate_scores

# Page Config
st.set_page_config(page_title="Real-time Productivity Dashboard", page_icon="🧠", layout="wide")


# FIREBASE INITIALIZATION 

def init_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            cred = credentials.Certificate("firebase-service-account.json")
            with open("firebase-service-account.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
                project_id = config.get("project_id", "emotion-productivity")
                bucket_name = f"{project_id}.appspot.com"
            
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            print(f"✅ Firebase initialized with project: {project_id}")
        except Exception as e:
            st.error(f"❌ Firebase init failed: {e}")
            st.stop()
    return firestore.client()


# MODEL & STATE INITIALIZATION


@st.cache_resource
def load_models():
    """Loads models with graceful fallback."""
    try:
        emotion_model = UncertaintyInference()
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None, None, None, None, None

    fatigue_model = FatigueCalculator()

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
                print("✅ Personalized model loaded from local storage")
            except Exception as e:
                print(f"⚠️ Could not load local model: {e}")

    return emotion_model, fatigue_model, prod_model, voice_analyzer, personalized_model

# Initialize Firebase
if 'db' not in st.session_state:
    st.session_state.db = init_firebase()

emotion_model, fatigue_model, prod_model, voice_analyzer, personalized_model = load_models()

if emotion_model is None:
    st.error("Application cannot start without emotion model.")
    st.stop()

# Initialize session state
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


# FEATURE ENGINEERING 

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

# DASHBOARD PAGES

def render_dashboard_page():
    st.title("🧠 Real-time Productivity Dashboard")

    # Calibration
    if not st.session_state.is_calibrated:
        st.warning("⚠️ Calibration required for accurate fatigue detection.")
        if st.button("⏱️ Start 20-Second Calibration", type="primary"):
            with st.spinner("Calibrating... Maintain neutral expression and look at camera."):
                fatigue_model.start_calibration()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    st.error("Cannot access webcam.")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                while fatigue_model.is_calibrating:
                    ret, frame = cap.read()
                    if not ret: break
                    fatigue_model.update_metrics(frame)
                    elapsed = time.time() - fatigue_model.calibration_start_time
                    progress = min(1.0, elapsed / fatigue_model.CALIBRATION_DURATION)
                    progress_bar.progress(progress)
                    status_text.text(f"Calibrating: {int(progress*100)}%")
                
                cap.release()
                st.session_state.is_calibrated = True
                st.success("✅ Calibration Complete! You may now start the webcam.")
                st.rerun()
        return

    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live Camera Feed")
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Webcam", use_container_width=True):
                st.session_state.webcam_running = True
        with col_stop:
            if st.button("⏹️ Stop Webcam", use_container_width=True):
                st.session_state.webcam_running = False
        
        # --- FOCUS SESSION CONTROLS ---
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
                        prod_scores = [d['productivity'] for d in st.session_state.focus_data]
                        high_focus_count = sum(1 for p in prod_scores if p >= 4.0)
                        focus_percent = (high_focus_count / len(prod_scores)) * 100
                        st.session_state.focus_session_score = focus_percent
                        try:
                            st.toast(f"🎉 Session Complete! You maintained high focus {focus_percent:.1f}% of the time.", icon="🎉")
                        except:
                            pass  # Ignore if Streamlit is gone
                else:
                    remaining = max(0, 25*60 - elapsed)
                    mins, secs = divmod(int(remaining), 60)
                    st.metric("Time Remaining", f"{mins:02d}:{secs:02d}")
        
        FRAME_WINDOW = st.image([])
        st.header("📈 Trend Analysis (Last 10 Minutes)")
        CHART_WINDOW = st.empty()

    with col2:
        st.header("📊 Real-time Metrics")
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

    # Main loop
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam.")
            st.session_state.webcam_running = False
            return

        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Webcam feed interrupted.")
                    break

                # Process frame
                emotion_data = emotion_model.predict_with_uncertainty(frame)
                fatigue_data = fatigue_model.update_metrics(frame)
                fatigue_index = fatigue_model.get_fatigue_index()
                final_scores = calculate_scores(emotion_data, fatigue_index)
                display_productivity = "⏳ Calculating..."

                if emotion_data:
                    # Buffer data
                    current_data = {
                        'timestamp': time.time(),
                        'confidence': emotion_data['confidence'],
                        'entropy': emotion_data['entropy'],
                        'valence': final_scores['valence'],
                        'arousal': final_scores['arousal'],
                        'EAR': fatigue_data['ear'],
                        'MAR': fatigue_data['mar'],
                        'FI': fatigue_index,
                        'top1': emotion_data['top1']
                    }
                    st.session_state.data_buffer.append(current_data)

                    # Predict every 5 seconds
                    current_time = time.time()
                    model_to_use = personalized_model or prod_model
                    if model_to_use and (current_time - st.session_state.last_pred_time > 5):
                        features = engineer_realtime_features(
                            st.session_state.data_buffer, 
                            st.session_state.prod_model_features if prod_model else ['Valence', 'Arousal', 'Fatigue Index']
                        )
                        if features is not None:
                            prediction = model_to_use.predict(features)[0]
                            display_productivity = round(float(prediction), 2)
                            st.session_state.last_pred_time = current_time

                            # Update history
                            new_row = pd.DataFrame([{
                                'Time': datetime.now(),
                                'Valence': final_scores['valence'],
                                'Arousal': final_scores['arousal'],
                                'Fatigue Index': fatigue_index,
                                'Productivity Score': display_productivity
                            }])
                            st.session_state.history_buffer = pd.concat([
                                st.session_state.history_buffer, new_row
                            ], ignore_index=True)

                            # Keep only last 10 minutes
                            cutoff = datetime.now() - pd.Timedelta(minutes=10)
                            st.session_state.history_buffer = st.session_state.history_buffer[
                                st.session_state.history_buffer['Time'] > cutoff
                            ]
                    else:
                        # Convert rule-based [-1,1] → [1,5]
                        rule_prod = final_scores['productivity']
                        display_productivity = round((rule_prod + 1) * 2 + 1, 2)  # Maps: -1→1, 0→3, 1→5
                        display_productivity = np.clip(display_productivity, 1.0, 5.0)

                    # Update UI
                    conf_val = emotion_data['confidence']
                    confidence_placeholder.metric(
                        "🎯 Emotion Confidence", 
                        f"{conf_val*100:.1f}%", 
                        help=f"Detected: {emotion_data['top1'].capitalize()}"
                    )

                    ear_placeholder.metric("👁️ Eye Aspect Ratio", f"{fatigue_data['ear']:.3f}")
                    mar_placeholder.metric("👄 Mouth Aspect Ratio", f"{fatigue_data['mar']:.3f}")
                    
                    # Color-code fatigue
                    fatigue_color = "green" if fatigue_index < 0.3 else "orange" if fatigue_index < 0.6 else "red"
                    fatigue_placeholder.markdown(
                        f"<h3 style='margin:0;'>💤 Fatigue Index</h3><h2 style='color:{fatigue_color}; margin:0;'>{fatigue_index:.2f}</h2>",
                        unsafe_allow_html=True
                    )

                    valence_placeholder.metric("😊 Valence (Mood)", f"{final_scores['valence']:.2f}")
                    arousal_placeholder.metric("⚡ Arousal (Energy)", f"{final_scores['arousal']:.2f}")
                    productivity_placeholder.metric("📊 Productivity Score", f"{display_productivity}/5.0")

                    # REAL-TIME ALERTS
                    try:
                        if fatigue_index > 0.7 and not st.session_state.get('fatigue_alert_shown', False):
                            st.toast("⚠️ Fatigue High — Take a Break!", icon="⚠️")
                            st.session_state.fatigue_alert_shown = True
                        elif fatigue_index <= 0.7:
                            st.session_state.fatigue_alert_shown = False

                        if final_scores['productivity'] > 0.5 and not st.session_state.get('focus_alert_shown', False):
                            st.toast("✅ Peak Focus — Tackle Hard Task Now!", icon="✅")
                            st.session_state.focus_alert_shown = True
                        elif final_scores['productivity'] <= 0.5:
                            st.session_state.focus_alert_shown = False
                    except:
                        pass  # Ignore if Streamlit connection is lost

                    # RECORD FOCUS DATA
                    if st.session_state.get('focus_session_active', False):
                        elapsed_focus = time.time() - st.session_state.focus_session_start
                        if elapsed_focus <= 25*60:
                            st.session_state.focus_data.append({
                                'time': time.time(),
                                'productivity': display_productivity if isinstance(display_productivity, float) else 3.0,
                                'fatigue': fatigue_index
                            })

                # Update chart
                if not st.session_state.history_buffer.empty:
                    chart_df = st.session_state.history_buffer.set_index('Time')
                    CHART_WINDOW.line_chart(
                        chart_df[['Valence', 'Arousal', 'Fatigue Index', 'Productivity Score']],
                        color=["#FF4B4B", "#4B4BFF", "#FFA500", "#2E8B57"]
                    )

                # Display frame
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        finally:
            cap.release()
            st.session_state.webcam_running = False
            st.info("⏹️ Webcam stopped. Click 'Start Webcam' to resume.")


# SESSION SUMMARY PAGE

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

    # Charts
    st.header("📈 Productivity & Fatigue Over Time")
    st.line_chart(df.set_index('Time')[['Productivity Score', 'Fatigue Index']], 
                  color=["#2E8B57", "#FFA500"])

    # Peak/Low Periods
    st.header("⏱️ Peak & Low Performance Periods")
    peak_time = df.loc[df['Productivity Score'].idxmax(), 'Time']
    low_time = df.loc[df['Productivity Score'].idxmin(), 'Time']
    st.write(f"✅ **Peak Productivity**: {peak_time.strftime('%H:%M:%S')} ({peak_prod:.2f}/5.0)")
    st.write(f"⚠️ **Low Productivity**: {low_time.strftime('%H:%M:%S')} ({df['Productivity Score'].min():.2f}/5.0)")

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
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Productivity Session Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Average Productivity: {avg_prod:.2f}/5.0", ln=True)
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

        # Daily Patterns
        st.header("⏰ Productivity by Hour of Day")
        hourly_avg = df.groupby('Hour')['Productivity Score'].mean()
        st.line_chart(hourly_avg)

        peak_hour = hourly_avg.idxmax()
        st.write(f"✅ **Your Peak Productivity Hour**: {peak_hour}:00")

        # Weekly Patterns
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
    
    # REMOVED: Personalized model button from sidebar
    
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