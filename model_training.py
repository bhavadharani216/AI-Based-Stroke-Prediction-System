import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_stroke_data.csv")

# ---- Remove Age feature ----
feature_cols = [
    "gender",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

X = df[feature_cols]
y = df["stroke"]

print("Features used:", feature_cols)
print("Class distribution before SMOTE:")
print(y.value_counts())

# ---- Handle imbalance ----
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
