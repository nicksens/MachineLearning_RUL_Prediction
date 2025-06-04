import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go # Import Plotly
import numpy as np # For generating sample data for the graph

# Load model that has been trained
model_pipeline = joblib.load("rul_model_pipeline.joblib")

# Page config
st.set_page_config(page_title="Battery RUL Predictor", layout="centered", initial_sidebar_state="auto")

# --- Styling ---
st.markdown("""
<style>
    /* ... (your existing styles remain here) ... */
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

# --- RUL Categories and Colors ---
RUL_CATEGORIES = {
    "Excellent RUL (Near New/Prime)": {"min": 1001, "max": float('inf'), "color": "rgba(76, 175, 80, 0.3)", "line_color": "green"}, # Light Green
    "Good RUL (Healthy Condition)": {"min": 751, "max": 1000, "color": "rgba(139, 195, 74, 0.3)", "line_color": "limegreen"},    # Lime Green
    "Medium RUL (Monitoring Recommended)": {"min": 401, "max": 750, "color": "rgba(255, 235, 59, 0.3)", "line_color": "gold"}, # Yellow
    "Low RUL (Replacement Planning Needed)": {"min": 151, "max": 400, "color": "rgba(255, 152, 0, 0.3)", "line_color": "orange"}, # Orange
    "Very Low RUL (Immediate Replacement Needed)": {"min": 0, "max": 150, "color": "rgba(244, 67, 54, 0.3)", "line_color": "red"}     # Red
}
MAX_RUL_DISPLAY = 1200 # Max RUL for y-axis in graph
MAX_OPERATIONAL_CYCLES = 1200 # Max operational cycles for x-axis in graph

# --- Plotting Function ---
def create_rul_trend_graph(predicted_rul_value=None):
    fig = go.Figure()

    # 1. Add RUL category bands (horizontal rectangles)
    for category_name, props in RUL_CATEGORIES.items():
        fig.add_hrect(
            y0=props["min"], y1=min(props["max"], MAX_RUL_DISPLAY), # Cap at MAX_RUL_DISPLAY
            fillcolor=props["color"],
            layer="below",
            line_width=0,
            annotation_text=category_name.split("(")[0].strip(), # Short name for annotation
            annotation_position="outside right",
            annotation_font_size=10,
        )

    # 2. Generate a sample degradation curve
    # This is a hypothetical curve for illustration
    op_cycles = np.linspace(0, MAX_OPERATIONAL_CYCLES, 100)
    # Example: RUL starts high and decreases (can be linear or non-linear)
    # For a slightly more realistic curve, let's use a polynomial or exponential decay
    # Simple linear for now:
    # initial_rul_for_curve = MAX_RUL_DISPLAY * 0.95
    # rul_values_curve = initial_rul_for_curve - (op_cycles * (initial_rul_for_curve / MAX_OPERATIONAL_CYCLES))
    # A slightly curved decay:
    initial_rul_for_curve = MAX_RUL_DISPLAY * 0.98
    rul_values_curve = initial_rul_for_curve * (1 - (op_cycles / MAX_OPERATIONAL_CYCLES)**1.5)
    rul_values_curve = np.clip(rul_values_curve, 0, MAX_RUL_DISPLAY) # Ensure it doesn't go below 0

    fig.add_trace(go.Scatter(
        x=op_cycles,
        y=rul_values_curve,
        mode='lines',
        name='Typical RUL Degradation',
        line=dict(color='navy', width=3),
        hoverinfo='skip' # No hover for the general trend line itself
    ))

    # 3. Highlight the predicted RUL
    if predicted_rul_value is not None:
        # Find corresponding "operational cycle" for the predicted RUL on the curve for marker placement
        # This is an approximation for visual effect
        try:
            # Find the operational cycle on the *illustrative curve* that is closest to the predicted RUL
            idx_on_curve = (np.abs(rul_values_curve - predicted_rul_value)).argmin()
            op_cycle_for_predicted_rul = op_cycles[idx_on_curve]

            fig.add_trace(go.Scatter(
                x=[op_cycle_for_predicted_rul], # Plot as a point on the curve's x-axis
                y=[predicted_rul_value],
                mode='markers+text',
                name='Your Battery\'s Predicted RUL',
                marker=dict(color='black', size=12, symbol='star'),
                text=[f"Predicted: {predicted_rul_value} Cycles"],
                textposition="top right",
                hoverlabel=dict(bgcolor="white", font_size=12)
            ))
            # Add a horizontal line for the predicted RUL
            fig.add_hline(
                y=predicted_rul_value,
                line=dict(color="black", width=2, dash="dash"),
                annotation_text=f"Predicted RUL: {predicted_rul_value}",
                annotation_position="bottom right"
            )
        except Exception as e:
            # st.warning(f"Could not place predicted RUL marker on graph: {e}")
            pass # Silently fail if marker placement is problematic

    # 4. Layout and labels
    fig.update_layout(
        title_text='Illustrative RUL Degradation Trend & Categories',
        title_x=0.5,
        xaxis_title='Operational Cycles (Battery Age)',
        yaxis_title='Remaining Useful Life (RUL) - Cycles',
        yaxis_range=[0, MAX_RUL_DISPLAY],
        xaxis_range=[0, MAX_OPERATIONAL_CYCLES],
        legend_title_text='Legend',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        paper_bgcolor='rgba(245, 245, 245, 0.8)', # Light grey paper
        margin=dict(l=50, r=50, b=80, t=80, pad=4)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig


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
        <li><b>NMC (Nickel Manganese Cobalt Oxide):</b> ...</li>
        <li><b>LCO (Lithium Cobalt Oxide):</b> ...</li>
        <li><b>NMC-LCO Blend:</b> ...</li>
        <li><b>18650:</b> ...</li>
    </ul>
    <b>Common Applications for NMC-LCO 18650 (or similar blended cells):</b>
    <ul>
        <li>High-performance portable electronics</li>
        <li>Power tools</li>
        <li>Medical devices</li>
        <li>E-bikes and other light electric mobility solutions</li>
        <li>Drones and robotics</li>
    </ul>
    The RUL (Remaining Useful Life) prediction helps in understanding how much longer these batteries can optimally perform in their respective applications.
    </p>
    """, unsafe_allow_html=True) # Keep your existing content here

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

    max_possible_rul_for_progress = 1133 # Based on your dataset's max RUL
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

    # RUL Categories and Recommendations
    recommendation_title = ""
    recommendation_message = ""
    recommendation_type = "info" # default

    if 0 <= predicted_rul <= 150:
        recommendation_title = "🆘 Very Low RUL (Immediate Replacement Needed)"
        recommendation_message = "Battery is at its end-of-life. Performance is severely degraded; immediate replacement is critical to prevent operational issues. This is a red flag—your battery is dying."
        recommendation_type = "error"
    elif 151 <= predicted_rul <= 400:
        recommendation_title = "⚠️ Low RUL (Replacement Planning Needed)"
        recommendation_message = "Battery is still functional but suboptimal. Significant capacity reduction means shorter usage times. Requires serious attention; plan for replacement soon."
        recommendation_type = "warning"
    elif 401 <= predicted_rul <= 750:
        recommendation_title = "🧐 Medium RUL (Monitoring Recommended)"
        recommendation_message = "Battery is in a transition phase. While working well, signs of decline might appear. Regular monitoring is key; start considering future replacement."
        recommendation_type = "info"
    elif 751 <= predicted_rul <= 1000:
        recommendation_title = "✅ Good RUL (Healthy Condition)"
        recommendation_message = "Battery is generally healthy and performing well. No significant performance drop yet, sufficient capacity for most needs. This is the expected condition for regularly used batteries."
        recommendation_type = "success"
    elif predicted_rul >= 1001:
        recommendation_title = "🌟 Excellent RUL (Near New/Prime)"
        recommendation_message = "Battery is exceptionally healthy, close to new or in prime condition. It has a very long remaining lifespan with optimal performance. This is the best condition your battery can be in."
        recommendation_type = "success_strong" # A custom type, or use success
    
    if recommendation_type == "error":
        st.error(f"**{recommendation_title}**\n\n{recommendation_message}")
    elif recommendation_type == "warning":
        st.warning(f"**{recommendation_title}**\n\n{recommendation_message}")
    elif recommendation_type == "info":
        st.info(f"**{recommendation_title}**\n\n{recommendation_message}")
    elif recommendation_type == "success" or recommendation_type == "success_strong":
        # For "Excellent", you might want a slightly different shade of green or emphasis if possible
        # For now, st.success handles both well.
        st.success(f"**{recommendation_title}**\n\n{recommendation_message}")
    else:
        st.markdown("<p class='recommendation-text'>Could not determine RUL category. Please check the input values.</p>", unsafe_allow_html=True)

    # --- Display the RUL Trend Graph ---
    st.markdown("<p class='section-header' style='margin-top:30px;'>📊 Visual RUL Context</p>", unsafe_allow_html=True)
    st.plotly_chart(create_rul_trend_graph(predicted_rul), use_container_width=True)


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
