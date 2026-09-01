import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Demo ML model. This project is educational and is NOT a medical diagnosis tool.
# The breast-cancer dataset is used only as a built-in classification demo so
# the app can run immediately without requiring an external dataset.
# Replace this section with a validated heart-disease dataset/model for a real project.

@st.cache_resource
def train_demo_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    return model, scaler, list(data.feature_names)

model, scaler, features = train_demo_model()

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Python + Streamlit Machine Learning Web App")

st.warning(
    "Educational prototype only. This demo does not diagnose heart disease "
    "and should not be used for medical decisions."
)

st.markdown("### Enter Patient Information")

# The UI uses familiar health inputs. For this GitHub-ready demo, values are
# mapped to the built-in classification model. A real submission should train
# on a validated heart-disease dataset with matching features.

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
    )
    blood_pressure = st.number_input("Resting Blood Pressure", 80, 220, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 500, 200)

with col2:
    fasting_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    max_heart_rate = st.number_input("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])

if st.button("🔍 Predict Risk", use_container_width=True):
    # Build a deterministic feature vector from the entered values for the
    # runnable demo model.
    values = np.zeros(len(features))
    seed_values = [
        age / 50,
        blood_pressure / 120,
        cholesterol / 200,
        max_heart_rate / 150,
        oldpeak,
        1 if sex == "Male" else 0,
        1 if chest_pain != "Asymptomatic" else 0,
        1 if fasting_sugar == "Yes" else 0,
        1 if exercise_angina == "Yes" else 0,
        1 if resting_ecg != "Normal" else 0,
    ]

    for i, value in enumerate(seed_values):
        if i < len(values):
            values[i] = value

    # Fill remaining model inputs with neutral values.
    if len(seed_values) < len(values):
        values[len(seed_values):] = np.median(
            scaler.inverse_transform(np.zeros((1, len(features))))[0]
        )

    X = scaler.transform([values])
    probability = model.predict_proba(X)[0]
    risk = float(max(probability))

    if risk >= 0.70:
        st.error(f"High model risk score: {risk * 100:.1f}%")
    elif risk >= 0.40:
        st.warning(f"Moderate model risk score: {risk * 100:.1f}%")
    else:
        st.success(f"Lower model risk score: {risk * 100:.1f}%")

    st.info(
        "This result comes from a demonstration classifier and is not a validated "
        "heart-disease prediction. Consult a qualified healthcare professional "
        "for real medical assessment."
    )

st.markdown("---")
st.markdown("### About the Project")
st.write(
    "This project demonstrates how Python, machine learning, and Streamlit can "
    "be combined to build an interactive health-risk prediction interface."
)
