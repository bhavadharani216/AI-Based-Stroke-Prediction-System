import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_stroke_data.csv")

# ---- Select only the features that are used in app.py ----
feature_cols = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

X = df[feature_cols]          # features
y = df["stroke"]              # target

print("Features used for training:", feature_cols)
print("X shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
