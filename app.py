import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------- Page Config ----------
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- Load Model & Data ----------
model = joblib.load("model.pkl")
df = pd.read_csv("cleaned_stroke_data.csv")

# ---------- Styles ----------
st.markdown("""
<style>
button {
    background-color: #1e40af !important;  /* Blue */
    color: white !important;
    font-weight: bold;
}
button:hover {
    background-color: #1e3a8a !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h2 style='color:#1e40af;'>🧠 Stroke Prediction Info</h2>", unsafe_allow_html=True)

    st.markdown("<p style='color:red; font-weight:bold;'>Stroke Prevention Tips:</p>", unsafe_allow_html=True)
    st.markdown("""
<ul style="color:green; font-weight:600;">
<li>Maintain healthy blood pressure</li>
<li>Exercise regularly</li>
<li>Eat a balanced diet</li>
<li>Avoid smoking and alcohol</li>
<li>Monitor BMI and glucose levels</li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1e40af;'>Dataset Stats</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:green; font-weight:bold;'>Total Records: {df.shape[0]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:green; font-weight:bold;'>Stroke Cases: {df['stroke'].sum()}</p>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<h1 style='text-align:center; color:#1e40af;'>🧠 Stroke Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #1e40af'>", unsafe_allow_html=True)

# ---------- Input Section ----------
st.markdown("<h2 style='color:#1e40af;'>Patient Information</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p style='color:green; font-weight:bold;'>Gender</p>", unsafe_allow_html=True)
    gender = st.selectbox("", ["Male", "Female", "Other"], key="gender", label_visibility="collapsed")

    st.markdown("<p style='color:green; font-weight:bold;'>Hypertension (0 = No, 1 = Yes)</p>", unsafe_allow_html=True)
    hypertension = st.selectbox("", [0, 1], key="hypertension", label_visibility="collapsed")

    st.markdown("<p style='color:green; font-weight:bold;'>Heart Disease (0 = No, 1 = Yes)</p>", unsafe_allow_html=True)
    heart_disease = st.selectbox("", [0, 1], key="heart", label_visibility="collapsed")

with col2:
    st.markdown("<p style='color:green; font-weight:bold;'>Average Glucose Level</p>", unsafe_allow_html=True)
    avg_glucose_level = st.number_input("", 50.0, 300.0, 100.0, key="glucose", label_visibility="collapsed")

    st.markdown("<p style='color:green; font-weight:bold;'>BMI</p>", unsafe_allow_html=True)
    bmi = st.number_input("", 10.0, 60.0, 20.0, key="bmi", label_visibility="collapsed")

    st.markdown("<p style='color:green; font-weight:bold;'>Smoking Status</p>", unsafe_allow_html=True)
    smoking_status = st.selectbox("", ["formerly smoked", "never smoked", "smokes", "Unknown"],
                                  key="smoking", label_visibility="collapsed")

# ---------- Encode ----------
gender_map = {"Male": 1, "Female": 0, "Other": 2}
smoking_map = {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}

gender = gender_map[gender]
smoking_status = smoking_map[smoking_status]

# ---------- Predict ----------
if st.button("Predict"):
    input_data = np.array([[gender, hypertension, heart_disease,
                            avg_glucose_level, bmi, smoking_status]])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.markdown(f"<h3 style='text-align:center; color:#1e40af;'>Probability of Stroke: {proba*100:.2f}%</h3>",
                unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("<h2 style='text-align:center; color:red;'>⚠️ High Risk of Stroke!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align:center; color:green;'>✅ Low Risk of Stroke!</h2>", unsafe_allow_html=True)

    st.subheader("📊 Your Stats vs Dataset Average")
    stats = pd.DataFrame({
        "Metric": ["BMI", "Average Glucose Level"],
        "Your Value": [bmi, avg_glucose_level],
        "Dataset Average": [df["bmi"].mean(), df["avg_glucose_level"].mean()]
    })

    fig, ax = plt.subplots()
    x = np.arange(len(stats["Metric"]))
    width = 0.35
    ax.bar(x - width/2, stats["Your Value"], width, label="Your Value")
    ax.bar(x + width/2, stats["Dataset Average"], width, label="Dataset Avg")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["Metric"])
    ax.set_ylabel("Value")
    ax.set_title("Patient vs Dataset Comparison")
    ax.legend()
    st.pyplot(fig)
