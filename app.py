import streamlit as st
import pandas as pd
import joblib

# Load model yang sudah ditrain
model_pipeline = joblib.load("rul_model_pipeline.joblib")

# Page config
st.set_page_config(page_title="Battery RUL Predictor", layout="centered")

# Header (Judul dan Penjelasan)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔋 Battery RUL Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Estimate the Remaining Useful Life (RUL) of a battery based on key voltage features.</p>", unsafe_allow_html=True)
st.markdown("---")

# Feature Labels dan value Default
feature_labels = {
    'Maximum Voltage Discharge (V)': 'F3',
    'Minimum Voltage Discharge (V)': 'F4',
    'Time at 4.15V (s)': 'F5'
}

features = list(feature_labels.keys())
default_values = {
    'Maximum Voltage Discharge (V)': 0.0,
    'Minimum Voltage Discharge (V)': 0.0,
    'Time at 4.15V (s)': 0
}

# Input data
with st.form(key='input_form'):
    st.markdown("<h4 style='color:#4CAF50;'>🧪 Input Battery Features</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        max_volt = st.number_input("🔌 Maximum Voltage Discharge (V)", value=default_values['Maximum Voltage Discharge (V)'], format="%.2f")
        time_415v = st.number_input("⏱️ Time at 4.15V (s)", value=default_values['Time at 4.15V (s)'], format="%d")
    with col2:
        min_volt = st.number_input("🔋 Minimum Voltage Discharge (V)", value=default_values['Minimum Voltage Discharge (V)'], format="%.2f")
    
    submit = st.form_submit_button("🔍 Predict RUL")

# Model Prediction
if submit:
    model_inputs = {
        'F3': max_volt,
        'F4': min_volt,
        'F5': time_415v
    }
    input_df = pd.DataFrame([model_inputs])

    prediction = model_pipeline.predict(input_df)

    st.markdown("---")
    st.markdown("<h4 style='color:#4CAF50;'>📈 Prediction Result</h4>", unsafe_allow_html=True)
    st.success(f"🔧 **Predicted Remaining Useful Life:** `{int(prediction[0])}` cycles")

    st.progress(min(1.0, prediction[0] / 1133))

# Footer
st.markdown("---")
