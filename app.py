# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
# Assuming you put your transform_text function in a file named preprocessor.py
from preprocessor import transform_text 

app = Flask(__name__)

# --- Load the Model Components ---
# Make sure these files are in the same directory as app.py
loaded_model = joblib.load('final_classifier.pkl')
loaded_cv = joblib.load('final_count_vectorizer.pkl')
loaded_k_selector = joblib.load('final_feature_selector.pkl')

@app.route('/')
def home():
    # Render the HTML template for the homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form submission
    raw_message = request.form['message']
    
    # --- Start the Prediction Pipeline ---
    
    # 1. Clean the text
    cleaned_text = transform_text(raw_message)
    
    # 2. Vectorize the text (needs to be passed as a list)
    vectorized_text = loaded_cv.transform([cleaned_text])
    
    # 3. Select the features
    final_features = loaded_k_selector.transform(vectorized_text)
    
    # 4. Predict
    prediction = loaded_model.predict(final_features)[0]
    
    # 5. Format the result
    if prediction == 1:
        result = "SPAM Message Detected!"
    else:
        result = "Legitimate Message (NOT SPAM)"

    # Return the result to the HTML template
    return render_template('index.html', prediction_text=result)

