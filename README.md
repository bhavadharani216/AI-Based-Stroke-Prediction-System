# 🧠 AI-Based Stroke Prediction System

This project predicts the risk of stroke using Machine Learning techniques. It uses a Random Forest model trained on real medical data and provides real-time predictions through a Streamlit web application.

---

## 🚀 Features
- Predicts stroke risk based on health data
- Displays probability of stroke (percentage)
- User-friendly dashboard using Streamlit
- Visual comparison with dataset average (BMI, Age, Glucose)
- Stroke prevention tips included
- Fast predictions without high hardware requirement

---

## 🛠 Technologies Used
- Python
- Scikit-learn
- Streamlit
- Pandas & NumPy
- Matplotlib
- Joblib

---

## 📂 Project Structure

📂 AI-Based Stroke Prediction System
│
├── app.py                  → Streamlit web app interface for predictions
├── model_training.py       → Script to train the ML model and save model.pkl
├── model.pkl               → Trained Random Forest model file
├── cleaned_stroke_data.csv → Processed dataset used for training
├── requirements.txt        → List of necessary libraries to run the project
└── README.md               → Project documentation

---

## 📊 Dataset Source

Healthcare-dataset-stroke-data.csv — Kaggle  
(Structured medical data used for stroke prediction)