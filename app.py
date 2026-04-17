# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:39:24 2026

@author: srima
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
#import time
from dotenv import load_dotenv

# Import custom modules
from Data_Preparation_Pipeline import compute_price_deltas, compute_delta_summary
from Pricing_Model import train_linear, train_random_forest, train_xgboost
from Elasticity import get_model_raw_metrics
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Load API Keys
load_dotenv()

st.set_page_config(page_title="AI Pricing Intelligence", layout="wide")

# --- 1. DYNAMIC MARKET SIMULATION (ALIGNED WITH PIPELINE) ---
st.sidebar.header("Market Simulation Parameters")
samples = st.sidebar.slider("Sample Size (Transactions)", 100, 5000, 1000)
base_p = st.sidebar.slider("Average Market Price ($)", 10.0, 500.0, 100.0)
sensitivity = st.sidebar.slider("Simulated Price Sensitivity", 1.0, 100.0, 20.0)
noise_level = st.sidebar.slider("Market Randomness (Noise)", 0.1, 2.0, 0.5)

def generate_aligned_data(n, p_avg, s, noise):
    """Generates synthetic data with columns matching the Data Preparation Pipeline"""
    np.random.seed(42)
    own_price = np.random.uniform(p_avg * 0.8, p_avg * 1.2, n)
    comp_1 = np.random.uniform(p_avg * 0.8, p_avg * 1.2, n)
    comp_2 = np.random.uniform(p_avg * 0.8, p_avg * 1.2, n)
    
    # Logic: Volume decreases as our price exceeds the competitor average
    avg_comp = (comp_1 + comp_2) / 2
    delta = own_price - avg_comp
    volume = 1000 - (s * delta) + np.random.normal(0, 50 * noise, n)
    
    df_raw = pd.DataFrame({
        "own_price": own_price,
        "comp_price_1": comp_1,
        "comp_price_2": comp_2,
        "volume": np.maximum(volume, 10)
    })
    
    # Run pipeline logic to create features like 'delta_median'
    df_processed = compute_price_deltas(df_raw, "own_price", ["comp_price_1", "comp_price_2"])
    df_processed = compute_delta_summary(df_processed, ["comp_price_1", "comp_price_2"])
    
    return df_processed

# Generate Data
df = generate_aligned_data(samples, base_p, sensitivity, noise_level)

st.title("🚀 AI Pricing Intelligence Assistant")
st.info("### Market Simulation Summary")
st.write("The simulation is ready. Below is the summary of the generated 'delta_median' (Price Gap vs Competitors) and Volume.")
st.dataframe(df[['own_price', 'comp_price_1', 'comp_price_2', 'delta_median', 'volume']].describe().T)

# User Verification Step
ready_to_proceed = st.checkbox("I have verified the simulation parameters. Proceed to Analysis.")

# --- 2. BACKEND ENGINE & STATUS TRACKING ---
if ready_to_proceed:
    st.divider()
    st.header("ML Model Analysis")
    selected_models = st.multiselect(
        "Select Models to Run", 
        ["Linear Regression", "XGBoost", "Random Forest"],
        default=["Linear Regression"]
    )
    
    if st.button("Run Global Analysis"):
        analysis_results = []
        log_messages = [] # For the downloadable log
        
        # Use st.status for a professional real-time log
        with st.status("Initializing Backend Analysis...", expanded=True) as status:
            # We use 'delta_median' as our feature based on your pipeline logic
            X = df[['delta_median']]
            y = df['volume']
            
            for model_name in selected_models:
                try:
                    msg = f"🔄 **{model_name}**: Training started..."
                    status.write(msg)
                    log_messages.append(msg)
                    
                    # Training Phase
                    if model_name == "Linear Regression": 
                        model = train_linear(X, y)
                    elif model_name == "XGBoost": 
                        model = train_xgboost(X, y)
                    else: 
                        model = train_random_forest(X, y)
                    
                    # Elasticity Extraction Phase
                    msg = f"📏 **{model_name}**: Extracting global slope (dQ/dP)..."
                    status.write(msg)
                    log_messages.append(msg)
                    
                    metrics = get_model_raw_metrics(model, X, price_col='delta_median')
                    
                    analysis_results.append({
                        "model": model_name,
                        "slope": metrics['raw_slope'],
                        "avg_p": metrics['avg_price'],
                        "avg_v": metrics['avg_volume'],
                        "r2": metrics['confidence']
                    })
                    
                    msg = f"✅ **{model_name}**: Analysis successfully completed."
                    status.write(msg)
                    log_messages.append(msg)
                    
                except Exception as e:
                    err_msg = f"❌ **{model_name} Failed**: {str(e)}"
                    status.write(err_msg)
                    log_messages.append(err_msg)
                    st.error(f"Issue detected with {model_name}. Check logs for details.")
            
            status.update(label="Analysis Process Finished", state="complete", expanded=False)

        # Store results for the AI Bot
        st.session_state['pricing_context'] = analysis_results
        st.session_state['execution_log'] = "\n".join(log_messages)

# --- 3. RESULTS DISPLAY & LOG DOWNLOAD ---
if 'pricing_context' in st.session_state and st.session_state['pricing_context']:
    st.subheader("Model Insights")
    cols = st.columns(len(st.session_state['pricing_context']))
    
    for i, res in enumerate(st.session_state['pricing_context']):
        with cols[i]:
            st.metric(label=f"{res['model']} Slope", value=f"{res['slope']:.2f}")
            st.caption(f"Confidence (R²): {res['r2']:.2%}")

    # Log Download Button
    st.download_button(
        label="Download Analysis Log",
        data=st.session_state['execution_log'],
        file_name="pricing_analysis_log.txt",
        mime="text/plain"
    )

# --- 4. AI STRATEGIST CHAT ---
if 'pricing_context' in st.session_state:
    st.divider()
    st.header("💬 AI Pricing Strategist")
    user_input = st.chat_input("Ask about the pragmatic impact of these slopes...")
    
    if user_input:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("API Key not found. Please add GOOGLE_API_KEY to your Streamlit Secrets or .env file.")
        else:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
                
                # Pragmatic System Prompt
                system_prompt = """
                You are a Pragmatic Pricing Strategist. You are provided with raw slope values (dQ/dP) from ML models.
                
                Instructions:
                1. Calculate Elasticity: E = Slope * (Avg_Price / Avg_Volume).
                2. Show the calculation for the user.
                3. Interpret the % change in volume for a 1% price increase.
                4. Compare the provided models and advise on price sensitivity (Elastic vs Inelastic).
                5. Use natural language to explain the business trade-off.
                """
                
                context = str(st.session_state['pricing_context'])
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Context: {context}\n\nUser Question: {user_input}")
                ]
                
                with st.spinner("AI is analyzing the market impact..."):
                    response = llm.invoke(messages)
                    st.chat_message("assistant").write(response.content)
            except Exception as e:
                st.error(f"AI Bot Error: {e}")