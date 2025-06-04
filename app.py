import streamlit as st
import pandas as pd
import joblib

# Load model that has been trained
model_pipeline = joblib.load("rul_model_pipeline.joblib")

# Page config
st.set_page_config(page_title="Battery RUL Predictor", layout="centered", initial_sidebar_state="auto")

# --- Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #2E8B57; /* Sea Green */
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        color: #555555;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #2E8B57; /* Sea Green */
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .recommendation-title {
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .recommendation-text {
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #3CB371; /* Medium Sea Green */
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57; /* Sea Green for progress bar */
    }
    .footer-attribution {
        text-align: center;
        color: #888888;
        font-size: 14px;
        margin-top: 30px;
    }
    .footer-attribution p {
        margin-bottom: 5px;
    }
    .footer-attribution ul {
        list-style-type: none;
        padding-left: 0;
        margin-top: 5px;
    }
    .footer-attribution li {
        font-size: 12px;
        color: #AAAAAA;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<p class='main-header'>🔋 Battery RUL Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Estimate the Remaining Useful Life (RUL) of NMC-LCO 18650 batteries based on key voltage features.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Information about NMC-LCO 18650 Batteries ---
with st.expander("ℹ️ About NMC-LCO 18650 Batteries & Their Applications", expanded=False):
    st.markdown("""
    <p style='font-size: 16px;'>
    NMC-LCO 18650 refers to a specific type of lithium-ion battery cell:
    <ul>
        <li><b>NMC (Nickel Manganese Cobalt Oxide):</b> A cathode chemistry known for its good balance of energy density, power output, and cycle life. It's widely used in electric vehicles, power tools, and medical devices.</li>
        <li><b>LCO (Lithium Cobalt Oxide):</b> Another cathode chemistry, traditionally used in consumer electronics (like smartphones and laptops) due to its high specific energy. However, it typically has a shorter cycle life and lower thermal stability than NMC.</li>
        <li><b>NMC-LCO Blend:</b> A cell designated as "NMC-LCO" likely indicates a blended cathode material, aiming to combine the high energy density benefits of LCO with the improved cycle life, power, and safety characteristics of NMC. This can make them suitable for applications requiring a good mix of these properties.</li>
        <li><b>18650:</b> This is a standard cylindrical cell format (18mm diameter, 65mm length).</li>
    </ul>
    <b>Common Applications for NMC-LCO 18650 (or similar blended cells):</b>
    Such cells, offering a balance of energy, power, and reasonable cycle life (like those with a 2.8Ah capacity and 1.5C discharge capability), can be used in:
    <ul>
        <li>High-performance portable electronics</li>
        <li>Power tools</li>
        <li>Medical devices</li>
        <li>E-bikes and other light electric mobility solutions</li>
        <li>Drones and robotics</li>
    </ul>
    The RUL (Remaining Useful Life) prediction helps in understanding how much longer these batteries can optimally perform in their respective applications.
    </p>
    """, unsafe_allow_html=True)

# --- Input Features ---
st.markdown("<p class='section-header'>🧪 Input Battery Features</p>", unsafe_allow_html=True)

feature_labels = {
    'Maximum Voltage Discharge (V)': 'F3',
    'Minimum Voltage Discharge (V)': 'F4',
    'Time at 4.15V (s)': 'F5'
}

default_values = {
    'Maximum Voltage Discharge (V)': 3.70,
    'Minimum Voltage Discharge (V)': 3.00,
    'Time at 4.15V (s)': 3600
}

with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    with col1:
        max_volt = st.number_input("⚡ Maximum Voltage Discharge (V)", value=default_values['Maximum Voltage Discharge (V)'], step=0.01, format="%.2f", help="Enter the maximum discharge voltage observed for the cycle.")
        time_415v = st.number_input("⏱️ Time at 4.15V (s) during charge", value=default_values['Time at 4.15V (s)'], step=100, format="%d", help="Enter the duration the battery spent at 4.15V during the constant voltage charging phase.")
    with col2:
        min_volt = st.number_input("📉 Minimum Voltage Discharge (V)", value=default_values['Minimum Voltage Discharge (V)'], step=0.01, format="%.2f", help="Enter the minimum discharge voltage (cut-off voltage) for the cycle.")

    submit = st.form_submit_button("🔍 Predict RUL")

# --- Prediction and Recommendation ---
if submit:
    model_inputs = {
        feature_labels['Maximum Voltage Discharge (V)']: max_volt,
        feature_labels['Minimum Voltage Discharge (V)']: min_volt,
        feature_labels['Time at 4.15V (s)']: time_415v
    }
    input_df = pd.DataFrame([model_inputs])

    prediction = model_pipeline.predict(input_df)
    predicted_rul = int(prediction[0])

    st.markdown("---")
    st.markdown("<p class='section-header'>📈 Prediction Result & Recommendation</p>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='font-size: 22px; text-align: center;'>Predicted RUL: <strong>{predicted_rul} cycles</strong></p>", unsafe_allow_html=True)

    max_possible_rul_for_progress = 1133
    progress_value = min(1.0, predicted_rul / max_possible_rul_for_progress)
    
    st.markdown(f"""
    <div style="margin-top: 10px; margin-bottom: 10px;">
        <div style="background-color: #e0e0e0; border-radius: 5px; padding: 3px;">
            <div style="background-color: #2E8B57; width: {progress_value*100}%; border-radius: 5px; text-align: center; color: white; font-weight: bold; padding: 2px 0;">
                {predicted_rul} / {max_possible_rul_for_progress} Cycles
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 0 <= predicted_rul <= 150:
        title = "🆘 Very Low RUL (Immediate Replacement Needed)"
        message = "Battery is at its end-of-life. Performance is severely degraded; immediate replacement is critical to prevent operational issues. This is a red flag—your battery is dying."
        st.error(f"**{title}**\n\n{message}")
    elif 151 <= predicted_rul <= 400:
        title = "⚠️ Low RUL (Replacement Planning Needed)"
        message = "Battery is still functional but suboptimal. Significant capacity reduction means shorter usage times. Requires serious attention; plan for replacement soon."
        st.warning(f"**{title}**\n\n{message}")
    elif 401 <= predicted_rul <= 750:
        title = "🧐 Medium RUL (Monitoring Recommended)"
        message = "Battery is in a transition phase. While working well, signs of decline might appear. Regular monitoring is key; start considering future replacement."
        st.info(f"**{title}**\n\n{message}")
    elif 751 <= predicted_rul <= 1000:
        title = "✅ Good RUL (Healthy Condition)"
        message = "Battery is generally healthy and performing well. No significant performance drop yet, sufficient capacity for most needs. This is the expected condition for regularly used batteries."
        st.success(f"**{title}**\n\n{message}")
    elif predicted_rul >= 1001:
        title = "🌟 Excellent RUL (Near New/Prime)"
        message = "Battery is exceptionally healthy, close to new or in prime condition. It has a very long remaining lifespan with optimal performance. This is the best condition your battery can be in."
        st.success(f"**{title}**\n\n{message}")
    else:
        st.markdown("<p class='recommendation-text'>Could not determine RUL category. Please check the input values.</p>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class='footer-attribution'>
    <p>Battery RUL Predictor | NMC-LCO 18650 Focus</p>
    <p>Created by: <strong>Kelompok 5 Machine Learning Class LM01</strong></p>
    <ul>
        <li>Nathanael Wilson Angelo - 2702212622</li>
        <li>Hernicksen Satria - 2702235235</li>
        <li>Rizky Fadhilah - 2702252696</li>
    </ul>
</div>
""", unsafe_allow_html=True)
