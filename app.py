# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import nltk
from preprocessor import transform_text 

# --- A. NLTK Resource Download Block (FIXES LookupError) ---
@st.cache_data(show_spinner="Preparing environment...")
def download_nltk_data():
    """Checks for and downloads required NLTK resources like 'punkt' and 'stopwords'."""
    resources = ['punkt', 'stopwords'] # Add 'wordnet' or others if used in preprocessor.py
    for resource in resources:
        try:
            # Try to find the resource first
            nltk.data.find(f'tokenizers/{resource}')
        except nltk.downloader.DownloadError:
            # If not found, download it
            nltk.download(resource, quiet=True)
        except LookupError:
             # Handle cases where the resource isn't found in expected paths
             nltk.download(resource, quiet=True)
    
download_nltk_data()
# -----------------------------------------------------------------


# --- B. Model Loading ---
@st.cache_data
def load_assets():
    """Loads all model components (Classifier, Vectorizer, Selector)."""
    try:
        loaded_model = joblib.load('final_classifier.pkl')
        loaded_cv = joblib.load('final_count_vectorizer.pkl')
        loaded_k_selector = joblib.load('final_feature_selector.pkl')
        return loaded_model, loaded_cv, loaded_k_selector
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: Required file not found. Ensure '{e.filename}' is uploaded.")
        return None, None, None

loaded_model, loaded_cv, loaded_k_selector = load_assets()


# --- C. UI Setup and Styling ---

st.set_page_config(
    page_title="Inbox Protector", 
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={'About': "A simple machine learning app for text classification."}
)

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.markdown(
        """
        <style>
        .title-text {
            font-size: 3em;
            font-weight: 900;
            color: #4f46e5;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title-text">@ Inbox Protector</div>', unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align: center; font-size: 1.1em; color: #555;'>
        Scan any suspected email or SMS text for malicious content. 🛡️
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---") 

# --- D. Input Form ---
st.subheader("✍️ Enter Message:")
raw_message = st.text_area(
    "Paste the full message content below.",
    placeholder="Example: 'Congratulations! You've been selected for a prize. Click this link: ...'",
    height=200,
    label_visibility="collapsed"
)

# --- E. Prediction Logic and Enhanced Output ---
st.markdown("<br>", unsafe_allow_html=True) 

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    scan_button = st.button('🚀 Run Security Scan', use_container_width=True)

if scan_button:
    if not loaded_model:
        st.error("❌ Cannot run scan: Model assets failed to load.")
    elif not raw_message:
        st.warning("⚠️ Please paste a message to begin the scan.")
    else:
        with st.spinner('🔍 Analyzing message content...'):
            # --- Prediction Pipeline ---
            cleaned_text = transform_text(raw_message)
            vectorized_text = loaded_cv.transform([cleaned_text])
            final_features = loaded_k_selector.transform(vectorized_)