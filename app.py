import streamlit as st 
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ================= STYLES =================
st.markdown("""
<style>
h1, h2, h3 { color: #1e40af !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #1e40af !important; }

section[data-testid="stSidebar"] strong { color: red !important; }
section[data-testid="stSidebar"] li { color: red !important; }

button[kind="primary"] {
    background-color: violet !important;
    color: black !important;
    font-weight: bold;
}
button[kind="primary"]:hover {
    background-color: #8b5cf6 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================= MAPPING =================
gender_map = {"Male": 1, "Female": 0, "Other": 2}
smoking_map = {
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3,
    "Unknown": 0
}

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
df = pd.read_csv("cleaned_stroke_data.csv")

# ================= BACKGROUND =================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1588776814546-c28d2d7a7b07?auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-position: center;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧠 Stroke Prediction Info")
    st.markdown("""
    **Stroke Prevention Tips:**  
    - Maintain healthy blood pressure  
    - Exercise regularly  
    - Eat a balanced diet  
    - Avoid smoking and alcohol  
    - Monitor BMI and glucose levels
    """)
    st.markdown("---")
    st.subheader("Dataset Stats")
    st.write("Total Records:", df.shape[0])
    st.write("Stroke Cases:", df['stroke'].sum())

# ================= HEADER =================
st.markdown("<h1 style='text-align: center;'>🧠 Stroke Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #1e40af'>", unsafe_allow_html=True)

# ================= INPUT FIELDS =================
with st.container():
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    with col2:
        avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 20.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ================= ENCODE =================
gender = gender_map[gender]
smoking_status = smoking_map[smoking_status]

# ================= PREDICTION =================
if st.button("Predict"):
    # Model expects: gender, hypertension, heart_disease, avg_glucose, bmi, smoking
    input_data = np.array([[gender, hypertension, heart_disease,
                            avg_glucose_level, bmi, smoking_status]])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.markdown(f"<h3 style='text-align: center;'>Probability of Stroke: {prediction_proba*100:.2f}%</h3>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>⚠️ High Risk of Stroke!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ Low Risk of Stroke!</h2>", unsafe_allow_html=True)

    # ================= GRAPH =================
    st.subheader("📊 Your Stats vs Dataset Average")
    stats = pd.DataFrame({
        "Metric": ["BMI", "Average Glucose Level"],
        "Your Value": [bmi, avg_glucose_level],
        "Dataset Average": [df["bmi"].mean(), df["avg_glucose_level"].mean()]
    })
    
    fig, ax = plt.subplots()
    x = np.arange(len(stats["Metric"]))
    width = 0.35
    ax.bar(x - width/2, stats["Your Value"], width, label='Your Value')
    ax.bar(x + width/2, stats["Dataset Average"], width, label='Dataset Avg')
    ax.set_xticks(x)
    ax.set_xticklabels(stats["Metric"])
    ax.set_ylabel("Value")
    ax.set_title("Patient vs Dataset Comparison")
    ax.legend()
    st.pyplot(fig)
