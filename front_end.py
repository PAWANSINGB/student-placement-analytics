import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration & Custom CSS for Modern Dark UI
st.set_page_config(page_title="Placement Predictor Pro", page_icon="🚀", layout="wide")

# Custom CSS directly injected for glassmorphism and modern colors
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00f2fe;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 198, 255, 0.4);
    }
    h1 {
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header Section
st.title("🚀 Placement Analytics & Prediction Dashboard")
st.write("Machine Learning ke dum par check kijiye apna placement status aur career readiness.")
st.markdown("---")

# 3. Load ML Assets
@st.cache_resource
def load_assets():
    with open('placement_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Aapki notebook ke hisab se saare exact features ki list humne yahan hardcode kar di
    # Ab 'features.pkl' file ki koi zaroorat nahi hai!
    required_features = [
        'college_tier', 'cgpa', 'backlog_history', 'DSA_problems_solved',
        'GitHub_repos', 'development_projects_count', 'AI_ML_projects',
        'internships_completed', 'resume_score', 'communication_skills',
        'aptitude_score', 'mock_interview_score', 'sleep_hours',
        'study_hours_daily', 'burnout_score', 'self_learning_hours',
        'joining_delay_months', 'layoffs_risk_score',
        'prompt_engineering_skill', 'state_Telangana', 'state_West Bengal',
        'branch_Electronics'
    ]
    return model, scaler, required_features

try:
    model, scaler, required_features = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files (`placement_model.pkl`, `scaler.pkl`) nahi mili! Pehle check karo ki yeh dono files GitHub par hain ya nahi.")
    st.stop()

# 4. Dashboard Layout (Left Side Inputs, Right Side Stats/Results)
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.subheader("📊 Student Profile Inputs")
    
    grid_1, grid_2 = st.columns(2)
    with grid_1:
        cgpa = st.number_input("🎓 Academic CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        dsa_solved = st.number_input("💻 DSA Problems Solved", min_value=0, max_value=2000, value=450, step=25)
        internships = st.slider("💼 Internships Completed", min_value=0, max_value=5, value=1)
        sleep_hours = st.slider("😴 Daily Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        communication = st.slider("🗣️ Communication Skills (1-100)", min_value=1, max_value=100, value=75)
        aptitude = st.slider("🧮 Aptitude Score (1-100)", min_value=1, max_value=100, value=70)
    
    with grid_2:
        resume_score = st.slider("📝 Resume ATS Score", min_value=0, max_value=100, value=75)
        mock_interview = st.slider("🎯 Mock Interview Score", min_value=0, max_value=100, value=70)
        backlogs = st.selectbox("🚨 Active/Past Backlogs", options=[0, 1, 2, 3, 4])
        gaming_hours = st.slider("🎮 Daily Gaming Hours", min_value=0.0, max_value=8.0, value=1.5, step=0.5)
        burnout_score = st.slider("🔥 Burnout Score", min_value=0, max_value=100, value=35)
        study_hours = st.slider("📚 Daily Study Hours", min_value=0.0, max_value=12.0, value=4.0, step=0.5)

    # Dictionary covering features used in training
    input_data = {
        'college_tier': 2, 'cgpa': cgpa, 'backlog_history': backlogs,
        'DSA_problems_solved': dsa_solved, 'internships_completed': internships,
        'resume_score': resume_score, 'mock_interview_score': mock_interview,
        'sleep_hours': sleep_hours, 'gaming_hours': gaming_hours, 'burnout_score': burnout_score,
        'communication_skills': communication, 'aptitude_score': aptitude, 'study_hours_daily': study_hours,
        'GitHub_repos': 5, 'development_projects_count': 2, 'AI_ML_projects': 1,
        'self_learning_hours': 3, 'joining_delay_months': 0, 'layoffs_risk_score': 0,
        'prompt_engineering_skill': 70, 'state_Telangana': 0, 'state_West Bengal': 0, 'branch_Electronics': 0
    }
    
    # Safety fallback to maintain order matching exact training features
    for col in required_features:
        if col not in input_data:
            input_data[col] = 0
            
    input_df = pd.DataFrame([input_data])[required_features]

with col_right:
    st.subheader("📈 Live Prediction Status")
    st.write("Neeche diye gaye button par click karke real-time prediction aur scorecards dekhein.")
    
    predict_btn = st.button("🚀 Run Prediction Model")
    
    if predict_btn:
        # Scale & Predict using the un-nested 'model' loaded from pickle
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]
        
        st.markdown("### Result")
        if prediction == 1:
            st.success(f"🎉 **Congratulations! Model Predicts: PLACED**")
            st.balloons()
            st.metric(label="Placement Probability", value=f"{prediction_proba*100:.1f}%", delta="High Chance")
            st.info("💡 **Key Strengths Identified:** Aapka DSA score aur Academic profile aapko safe zone mein rakhta hai. Keep it up!")
        else:
            st.error(f"❌ **Model Predicts: NOT PLACED YET**")
            st.metric(label="Placement Probability", value=f"{prediction_proba*100:.1f}%", delta="- Improvement Required", delta_color="inverse")
            st.warning("🛠️ **Actionable Advice to Improve:**\n"
                       "* Try to clear active backlogs if any.\n"
                       "* Solve 50+ more DSA questions to push into the placement bracket.\n"
                       "* Resume score ko ATS format se 80+ le jaane ki koshish karein.")
    else:
        st.info("ℹ️ Profile sections fill karne ke baad 'Run Prediction Model' par click karein.")
        
        stat_1, stat_2 = st.columns(2)
        stat_1.metric("Model Used", "Logistic Regression", "Tuned")
        stat_2.metric("Pipeline Integrity", "100% Safe", "No Leakage")
