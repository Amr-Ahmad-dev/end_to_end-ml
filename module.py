import streamlit as st
import pandas as pd
import numpy as np
import joblib as jp

# Set up page layout
st.set_page_config(page_title="Health Risk Assessment", layout="centered")

# Load saved pipeline artifact
@st.cache_resource
def load_pipeline():
    return jp.load("chronic_disease_model.joblib")

try:
    artifact = load_pipeline()
    model = artifact["model"]
    threshold = artifact.get("threshold", 0.35)
    expected_features = artifact["feature_columns"]
except Exception as e:
    st.error("Could not load 'chronic_disease_model.joblib'. Ensure the file is in your working directory.")
    st.stop()

st.title("Chronic Disease Risk Assessment")
st.write("Fill in your details below to check your estimated 10-year risk profile.")

# --- FORM INPUTS ---
with st.form("risk_form"):
    st.subheader("Key Health Indicators")
    
    # 1. General Health
    gen_hlth_labels = {
        "Unspecified / Skip": None,
        "1 - Excellent": 1,
        "2 - Very Good": 2,
        "3 - Good": 3,
        "4 - Fair": 4,
        "5 - Poor": 5
    }
    gen_hlth_choice = st.selectbox("General Health Assessment", list(gen_hlth_labels.keys()))
    
    # 2. High Blood Pressure
    bp_labels = {"Unspecified / Skip": None, "No": 0, "Yes": 1}
    bp_choice = st.selectbox("High Blood Pressure Diagnosis", list(bp_labels.keys()))
    
    # 3. High Cholesterol
    chol_labels = {"Unspecified / Skip": None, "No": 0, "Yes": 1}
    chol_choice = st.selectbox("High Cholesterol Diagnosis", list(chol_labels.keys()))
    
    # 4. Difficulty Walking
    diff_walk_labels = {"Unspecified / Skip": None, "No": 0, "Yes": 1}
    diff_walk_choice = st.selectbox("Difficulty Walking or Climbing Stairs", list(diff_walk_labels.keys()))
    
    # 5. BMI Input
    bmi_input = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=None, placeholder="e.g., 24.5")
    
    # 6. Age Category
    age_labels = {
        "Unspecified / Skip": None,
        "Age 18 - 24": 1,
        "Age 25 - 29": 2,
        "Age 30 - 34": 3,
        "Age 35 - 39": 4,
        "Age 40 - 44": 5,
        "Age 45 - 49": 6,
        "Age 50 - 54": 7,
        "Age 55 - 59": 8,
        "Age 60 - 64": 9,
        "Age 65 - 69": 10,
        "Age 70 - 74": 11,
        "Age 75 - 79": 12,
        "Age 80 or older": 13
    }
    age_choice = st.selectbox("Age Bracket", list(age_labels.keys()))
    
    # 7. Income Bracket
    income_labels = {
        "Unspecified / Skip": None,
        "Less than $10,000": 1,
        "$10,000 to $14,999": 2,
        "$15,000 to $19,999": 3,
        "$20,000 to $24,999": 4,
        "$25,000 to $34,999": 5,
        "$35,000 to $49,999": 6,
        "$50,000 to $74,999": 7,
        "$75,000 or more": 8
    }
    income_choice = st.selectbox("Annual Income Bracket", list(income_labels.keys()))
    
    # 8. Education Level
    edu_labels = {
        "Unspecified / Skip": None,
        "Never attended school or kindergarten only": 1,
        "Grades 1 through 8 (Elementary)": 2,
        "Grades 9 through 11 (Some High School)": 3,
        "Grade 12 or GED (High School Graduate)": 4,
        "College 1 to 3 years (Some College)": 5,
        "College 4 years or more (College Graduate)": 6
    }
    edu_choice = st.selectbox("Education Level", list(edu_labels.keys()))
    
    # Form submission button
    submitted = st.form_submit_button("Submit Assessment")

# --- DATA PROCESSING & INFERENCE ---
if submitted:
    # Population default values for unselected features
    default_values = {
        "GenHlth": 2,
        "HighBP": 0,
        "HighChol": 0,
        "DiffWalk": 0,
        "BMI": 27.0,
        "Age": 7,
        "Income": 6,
        "Education": 5,
        "MentHlth": 0,
        "PhysHlth": 0,
        "Smoker": 0,
        "Stroke": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "Sex": 0,
        "HeartDiseaseorAttack": 0
    }

    # Extract user inputs
    user_inputs = {
        "GenHlth": gen_hlth_labels[gen_hlth_choice],
        "HighBP": bp_labels[bp_choice],
        "HighChol": chol_labels[chol_choice],
        "DiffWalk": diff_walk_labels[diff_walk_choice],
        "BMI": bmi_input,
        "Age": age_labels[age_choice],
        "Income": income_labels[income_choice],
        "Education": edu_labels[edu_choice]
    }

    # Construct complete row using defaults where user skipped input
    row = {}
    for col in expected_features:
        if col in user_inputs and user_inputs[col] is not None:
            row[col] = user_inputs[col]
        elif col in default_values:
            row[col] = default_values[col]
        else:
            row[col] = 0

    # Build single-row DataFrame aligned to expected features
    input_df = pd.DataFrame([row])[expected_features]

    # Predict using saved pipeline (runs internal scaling automatically)
    probability = model.predict_proba(input_df)[0, 1]
    prediction = 1 if probability >= threshold else 0

    # Output results
    st.markdown("---")
    st.subheader("Assessment Results")
    st.write(f"Estimated Probability: **{probability * 100:.1f}%**")

    if prediction == 1:
        st.warning("Assessment Indicator: Higher risk profile detected.")
    else:
        st.success("Assessment Indicator: Lower risk profile detected.")

    # Lifestyle Suggestions
    st.subheader("General Recommendations")
    st.write("- **Nutrition**: Aim for balanced meals rich in whole grains, vegetables, and lean proteins.")
    st.write("- **Activity**: Try to get around 150 minutes of moderate physical activity every week.")
    st.write("- **Routine Checks**: Schedule regular check-ups to track blood pressure and glucose levels.")
    st.write("- **Sleep & Rest**: Aim for 7 to 8 hours of quality sleep to support recovery and reduce stress.")