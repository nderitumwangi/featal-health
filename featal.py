import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Load Model and Scaler ---
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Streamlit App UI ---
st.set_page_config(page_title="Fetal Health Prediction", page_icon="🩺", layout="centered")
st.title("🧠 Fetal Health Classification using CTG Data")
st.write("""
This application predicts **Fetal Health Status** (Normal, Suspect, or Pathological)  
based on **Cardiotocography (CTG)** measurements.
""")

# --- Define input features ---
feature_names = [
    'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
    'histogram_median', 'histogram_variance', 'histogram_tendency'
]

# --- Create input form ---
st.sidebar.header("🧩 Enter CTG Parameters")
input_data = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature.replace('_',' ').capitalize()}", value=0.0, step=0.01, format="%.4f")
    input_data.append(val)

# --- Predict Button ---
if st.button("🔍 Predict Fetal Health"):
    # Preprocess input
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    # --- Output interpretation ---
    labels = {1: "🟢 Normal", 2: "🟡 Suspect", 3: "🔴 Pathological"}
    st.subheader("📊 Prediction Result")
    st.success(f"**Predicted Fetal Health:** {labels[prediction]}")
    
    st.write("### Confidence Levels:")
    st.write(f"- 🟢 Normal: {prob[0]*100:.2f}%")
    st.write(f"- 🟡 Suspect: {prob[1]*100:.2f}%")
    st.write(f"- 🔴 Pathological: {prob[2]*100:.2f}%")

    # Optional: Show feature impact
    st.write("---")
    st.caption("Model used: Gradient Boosting Classifier (Macro F1: 0.8779)")

else:
    st.info("👈 Enter feature values on the left and click **Predict Fetal Health**.")
