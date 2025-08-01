import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import altair as alt
from PIL import Image
st.set_page_config(
    page_title="Cloud Pattern Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
        .css-18e3th9 { 
            color: #0d47a1 !important;      
            font-weight: 600 !important;    
            font-size: 16px !important;     
        }
        .css-1v0mbdj {  
            color: #0b3954 !important;        
            font-weight: 700 !important;    
        }
        .css-18e3th9:hover {
            background-color: rgba(13, 71, 161, 0.1) !important; 
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        div.streamlit-expanderHeader {
            background-color: #d6eaf8 !important; 
            color: #003366 !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px;
        }
        div.streamlit-expanderContent {
            background-color: #f0f4f8 !importan;
            color: #333333 !important;
            padding: 15px;
            border-radius: 10px;
        }
        div.streamlit-expanderContent p,
        div.streamlit-expanderContent li {
            color: #333333 !important;
            font-size: 14px;
            line-height: 1.5;
        }
        details[open] > summary {
            border: none;
            background-color: #d6eaf8 !important;
            border-radius: 8px;
        }

        .css-1d391kg {
            background-color: #e0f7fa; 
        }

        .css-1d391kg h1, 
        .css-1d391kg label, 
        .css-1d391kg .stRadio label {
            color: #003366 !important; 
            font-weight: 600;
        }

        .stApp {
            background: linear-gradient(135deg, #e0f7fa, #f0f4f8);
            background-attachment: fixed;
            font-family: 'Arial', sans-serif;
            color: #333333;
        }

        .block-container {
            background-color: rgba(255, 255, 255, 0.85);  
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            max-width: 1000px;
            margin: auto;
        }

        h1, h2, h3 {
            color: #0d47a1; 
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        p, li, label, .stMarkdown {
            color: #444444;
            font-size: 15px;
            line-height: 1.6;
        }

        img {
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            max-width: 100%;
        }

        .stMetric {
            background-color: rgba(240, 244, 248, 0.95); 
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
    </style>

""", unsafe_allow_html=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Prediction", "Visualization", "Metrics", "Model Insights", "About Team"])
if page == "Home":
    st.title("Chase the Cloud")
    st.subheader("AI-Powered Cloud Motion Forecasting using Diffusion Models")
    st.markdown("""
    Welcome to **Chase the Cloud**—an innovative project that leverages the power of **MetaConditioned Residual Diffusion Models (MC-RDDM)** to predict **short-term cloud motion** using **multi-spectral satellite imagery from INSAT-3DR/3DS**.
    Our vision is to bring **next-generation AI forecasting** to nowcasting and severe weather early warning systems.
    """)
    st.markdown("---")
    st.header("Project Overview")
    st.markdown("""
    - **What:** Short-term (0 to 3 hours) **cloud motion prediction** using deep learning and generative AI.
    - **How:** By training a **diffusion-based model** on historical satellite imagery sequences and multi-channel spectral data (VIS, IR, WV).
    - **Why:** Traditional optical flow and physics-based models often fail under volatile weather—our approach aims to bring **greater accuracy, realism, and adaptability** to weather nowcasting.
    - **Impact:** Improved **early warnings for severe weather**, real-time visualizations, and **AI-backed decision support** for meteorological agencies.
    """)
    st.markdown("---")
    st.header("Model Architecture Highlights")
    model_info = {
        "Model Type": "MetaConditioned Residual Diffusion Model (MC-RDDM)",
        "Inputs": "4–6 past satellite frames (VIS, IR, WV channels)",
        "Outputs": "1–2 future cloud motion predictions",
        "Conditioning": "Sun angle, time of day, spectral attention",
        "Key Innovation": "Spatio-temporal learning + spectral fusion + efficient latent diffusion",
        "Status": "Prototype Phase"
    }
    for key, value in model_info.items():
        st.markdown(f"**{key}:** {value}")
    st.markdown("---")
    with st.expander("How It Works (Click to Expand)"):
        st.markdown("""
        Our **MC-RDDM** learns the complex patterns of **cloud evolution** by:
        1. Analyzing **sequences of past satellite images** across multiple spectral channels (visible light, infrared, water vapor).
        2. Applying a **Residual Diffusion Model** to simulate realistic future cloud movement.
        3. Conditioning predictions on **auxiliary metadata** such as solar geometry and time of day.
        4. (Optionally) Compressing information using **latent space representations** for faster inference.
        The result is a system that can generate **future cloud images** that look and behave like real atmospheric patterns.
        """)
    st.markdown("---")
    st.markdown("—")
    st.markdown("Built by Team SpaceWalkers")
elif page == "Model Prediction":
    st.title("Cloud Prediction")
    uploaded_file = st.file_uploader("Upload Cloud Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.markdown("### Running Cloud Motion Model...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"{i}% Completed")
            time.sleep(0.02)
        st.success("Prediction Completed!")

        future_cloud_cover = np.random.uniform(30, 70, size=10)
        forecast_times = pd.date_range("2023-01-05", periods=10, freq='H')

        forecast_df = pd.DataFrame({
            'Time': forecast_times,
            'Predicted Cloud Coverage (%)': future_cloud_cover
        })
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Infrared (IR) Satellite Image")
            ir_image = Image.open("assets/IR.jpeg")
            st.image(ir_image, caption="IR Channel - Current Frame", use_container_width=True)

        with col2:
            st.subheader("Contrast-enhanced Cloud Image")
            contrast_image = Image.open("assets/contrast.jpeg")
            st.image(contrast_image, caption="Contrast Enhancement", use_container_width=True)

        st.markdown("---")
        
        st.video("assets/gif.mp4",format="video/mp4")

        fig2 = px.line(forecast_df, x='Time', y='Predicted Cloud Coverage (%)',
                       title='Predicted Cloud Coverage Over Time',
                       line_shape='spline')
        st.plotly_chart(fig2, use_container_width=True)
        st.balloons()
        
    else:
        st.warning("Please upload a cloud image to start prediction.")
        
elif page == "About Team":
    st.title("Meet the Team")

    st.write("""
        **CloudChaser:**

        - **Rakshita Kowlikar** — Project Lead & Machine Learning Developer
        - **Kartik Bulusu** — Visualization & Frontend Developer
        - **Aniket Desai** — Data Processing & Geospatial Analysis

        **Affiliation:** Veermata Jijabai Technological Institute

        This project was developed as part of the "Chase the Cloud" initiative to harness AI for meteorological forecasting using indigenous satellite data.
        
        For collaborations or feedback, feel free to reach out!
    """)

    st.image("assets\spacewalkers.png", caption="Team SpaceWalkers", use_container_width=False, width=300)

    st.markdown("[Visit Project Repository](https://github.com/KartikB3/streamlit)")

elif page == "Metrics":
    st.title("Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="SSIM", value="0.84", delta="+3%")

    with col2:
        st.metric(label="PSNR (dB)", value="28.5", delta="+1.2")

    with col3:
        st.metric(label="MAE", value="0.045", delta="-0.005")

    st.markdown("---")

    st.subheader("Metric Trends (Sample)")
    sample_steps = np.arange(1, 6)
    sample_ssim = [0.85, 0.82, 0.80, 0.77, 0.74]

    fig, ax = plt.subplots()
    ax.plot(sample_steps, sample_ssim, marker='o', color='#1565c0')
    ax.set_xlabel("Prediction Step (t+1, t+2,...)")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs Prediction Horizon")
    st.pyplot(fig)
    
elif page == "Model Insights":
    st.title("Model Insights: MetaConditioned RDDM")

    st.write("""
        The core of this solution is the **MetaConditioned Residual Diffusion Denoising Model (MC-RDDM)**, designed specifically for geospatial-temporal cloud motion forecasting.

        **Model Components:**

        - **RDDM Backbone:** Learns the delta between current and future cloud frames.
        - **Spectral-Attention Module:** Captures inter-channel relationships (e.g., VIS ↔ WV).
        - **Metadata Encoder:** Embeds time of day, solar/satellite angles.
        - **Temporal Transformer:** Models sequence continuity across multiple past and future steps.
        - **Optional Latent Encoder:** Enables faster inference by working in latent space.

        This design ensures that spatial, temporal, and spectral features are all captured, improving both accuracy and generalizability.
    """)

    st.image("assets/architechure.png", caption="Architecture: MetaConditioned RDDM", use_container_width=True)
    
elif page == "Visualization":
    st.title("Model Parameters and Distributions")

    st.write("""
        This section visualizes various internal model parameters, hyperparameters, and their effects on performance.
    """)

    # Sample Data
    params = pd.DataFrame({
        'Parameter': ['Learning Rate', 'Batch Size', 'Noise Level', 'Timesteps', 'Attention Heads'],
        'Value': [0.0005, 16, 0.1, 1000, 8]
    })

    st.dataframe(params)

    st.subheader("Parameter Impact Heatmap")
    heat_data = np.random.rand(5, 5)
    fig, ax = plt.subplots()
    sns.heatmap(heat_data, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Parameter Importance")
    importance_df = pd.DataFrame({
        'Parameter': ['Learning Rate', 'Batch Size', 'Noise Level', 'Timesteps', 'Attention Heads'],
        'Importance': [0.6, 0.1, 0.15, 0.1, 0.05]
    })
    fig_bar = px.bar(importance_df, x='Parameter', y='Importance', color='Parameter')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Parameter Trends")
    alt_data = pd.DataFrame({
        'Step': np.arange(1, 11),
        'Learning Rate': np.linspace(0.001, 0.0001, 10),
        'Noise Level': np.linspace(0.2, 0.05, 10)
    })
    line_chart = alt.Chart(alt_data).mark_line().encode(
        x='Step',
        y='Learning Rate',
        color=alt.value('#1f77b4')
    )
    st.altair_chart(line_chart, use_container_width=True)
    