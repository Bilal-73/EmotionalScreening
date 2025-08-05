import streamlit as st
import joblib
import numpy as np

# =========================
# Load Model & Encoders
# =========================
model = joblib.load("neuroticure_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# =========================
# App UI
# =========================
st.set_page_config(page_title="NeurotiCure Screening", page_icon="🧠", layout="centered")
st.title("🧠 NeurotiCure Emotional Screening")
st.write("Answer the following questions to get a mental health recommendation.")

# -------------------------
# Step 1: Basic Info
# -------------------------
name = st.text_input("Name", placeholder="Enter your name")
age = st.number_input("Age", min_value=10, max_value=100, value=25)

gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
gender_encoded = gender_encoder.transform([gender])[0]

# -------------------------
# Step 2: Emotional Screening Questions (Radio Buttons Inline)
# -------------------------
q4 = st.radio(
    "Q4: How have you been feeling emotionally over the past two weeks?",
    ["Good (0)", "Okay (1)", "Upset (2)", "Overwhelmed (3)"],
    horizontal=True
)
q4_val = int(q4.split("(")[1].replace(")", ""))

q5 = st.radio(
    "Q5: Have you had trouble sleeping, eating, or concentrating lately?",
    ["No (0)", "Yes (1)"],
    horizontal=True
)
q5_val = int(q5.split("(")[1].replace(")", ""))

q6 = st.radio(
    "Q6: Have you gone through anything stressful recently?",
    ["No (0)", "Yes (1)"],
    horizontal=True
)
q6_val = int(q6.split("(")[1].replace(")", ""))

q7 = st.radio(
    "Q7: Do you feel like your emotions are sometimes too much to manage?",
    ["Never (0)", "Sometimes (1)", "Often (2)"],
    horizontal=True
)
q7_val = int(q7.split("(")[1].replace(")", ""))

q8 = st.radio(
    "Q8: Do you have someone to talk to when you're feeling low or stressed?",
    ["Yes (0)", "No (1)"],
    horizontal=True
)
q8_val = int(q8.split("(")[1].replace(")", ""))

q9 = st.radio(
    "Q9: Would you like to explore what might be affecting your mental well-being?",
    ["Maybe later (0)", "Yes (1)"],
    horizontal=True
)
q9_val = int(q9.split("(")[1].replace(")", ""))

# -------------------------
# Step 3: Predict
# -------------------------
if st.button("🔍 Get Recommendation"):
    # Prepare input
    features = np.array([[age, gender_encoded, q4_val, q5_val, q6_val, q7_val, q8_val, q9_val]])
    
    # Predict
    prediction = model.predict(features)
    redirection = target_encoder.inverse_transform(prediction)[0]
    
    # Output
    st.subheader("Recommendation for " + (name if name else "User") + ":")
    st.success(f"**{redirection}**")
    
    # Optional: Show reasoning
    total_score = q4_val + q5_val + q6_val + q7_val + q8_val
    st.write(f"**Your distress score (Q4–Q8):** {total_score} / 8")
    st.write("Recommendation is based on your responses and our trained model.")
