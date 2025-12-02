import streamlit as st 
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Mapping dictionaries for categorical inputs
gender_map = {"Male": 1, "Female": 0, "Other": 2}
smoking_map = {
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3,
    "Unknown": 0
}

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load trained model and data
model = joblib.load("model.pkl")
df = pd.read_csv("cleaned_stroke_data.csv")

# Background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1588776814546-c28d2d7a7b07?auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Sidebar
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

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Stroke Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)

# Input fields (Removed Ever Married, Residence Type, Work Type)
with st.container():
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    with col2:
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=20.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode categorical inputs
gender = gender_map[gender]
smoking_status = smoking_map[smoking_status]

# Prediction button
if st.button("Predict"):
    # Adjusted input array (Removed Ever Married, Residence Type, Work Type)
    input_data = np.array([[gender, age, hypertension, heart_disease,
                            avg_glucose_level, bmi, smoking_status]])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of stroke
    
    # Display results
    st.markdown(f"<h3 style='text-align: center; color: white;'>Probability of Stroke: {prediction_proba*100:.2f}%</h3>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown("<h2 style='text-align: center; color: red;'>⚠️ High Risk of Stroke!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>✅ Low Risk of Stroke!</h2>", unsafe_allow_html=True)

    # Visualization: Compare patient BMI, Glucose, Age with dataset average
    st.subheader("📊 Your Stats vs Dataset Average")
    stats = pd.DataFrame({
        "Metric": ["BMI", "Average Glucose Level", "Age"],
        "Your Value": [bmi, avg_glucose_level, age],
        "Dataset Average": [df["bmi"].mean(), df["avg_glucose_level"].mean(), df["age"].mean()]
    })
    
    fig, ax = plt.subplots()
    x = np.arange(len(stats["Metric"]))
    width = 0.35
    ax.bar(x - width/2, stats["Your Value"], width, label='Your Value', color='skyblue')
    ax.bar(x + width/2, stats["Dataset Average"], width, label='Dataset Avg', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(stats["Metric"])
    ax.set_ylabel("Value")
    ax.set_title("Patient vs Dataset Comparison")
    ax.legend()
    st.pyplot(fig)
