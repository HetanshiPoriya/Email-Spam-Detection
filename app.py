# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from preprocessor import transform_text 

# --- 1. Load the Model Components EFFICIENTLY using Streamlit Caching ---
# Caching prevents the models from reloading every time the app updates, improving performance.
@st.cache_data
def load_assets():
    try:
        # Load all three necessary model/vectorizer assets
        loaded_model = joblib.load('final_classifier.pkl')
        loaded_cv = joblib.load('final_count_vectorizer.pkl')
        loaded_k_selector = joblib.load('final_feature_selector.pkl')
        return loaded_model, loaded_cv, loaded_k_selector
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Ensure '{e.filename}' is uploaded.")
        return None, None, None

loaded_model, loaded_cv, loaded_k_selector = load_assets()

# --- 2. UI Setup (Replaces Flask/HTML Header and Styling) ---
st.set_page_config(page_title="Secure Mail Scanner", layout="centered")

# Use st.markdown and unsafe_allow_html=True to replicate basic Tailwind styles
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        font-weight: 800;
        color: #4f46e5; /* Tailwind indigo-700 */
        text-align: center;
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-header">📧 Inbox Protector</h1>', unsafe_allow_html=True)
st.markdown("Check any suspected email or message for spam.")
st.markdown("---") 

# --- 3. Input Form (Replaces HTML <form> and <textarea>) ---
raw_message = st.text_area(
    "Enter Message:",
    placeholder="Example: 'You have won a free iPhone! Click this survey link to claim your prize.'",
    height=200
)

# --- 4. Prediction Logic (Replaces the Flask @app.route('/predict') function) ---
if st.button('Run Security Scan'):
    if not loaded_model:
        # This handles the case if a model failed to load in load_assets()
        st.error("Cannot proceed: Model assets failed to load.")
    elif not raw_message:
        st.warning("Please enter a message to scan.")
    else:
        # Create a spinner to show activity while processing (replaces animation)
        with st.spinner('Analyzing message...'):
            # 1. Clean the text
            cleaned_text = transform_text(raw_message)

            # 2. Vectorize the text
            # Ensure input is a list for the transformer
            vectorized_text = loaded_cv.transform([cleaned_text])

            # 3. Select the features
            final_features = loaded_k_selector.transform(vectorized_text)

            # 4. Predict
            prediction = loaded_model.predict(final_features)[0]

        # 5. Format and Display the result (Replaces Jinja conditional rendering)
        st.markdown("## Scan Results:")
        
        if prediction == 1:
            st.error("🚫 **DANGER! SPAM Message Detected!**")
        else:
            st.success("✅ **CLEAN! Legitimate Message (NOT SPAM)**")