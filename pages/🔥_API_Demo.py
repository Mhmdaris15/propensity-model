import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
# input file
model = joblib.load('model/Supermodel API Pre-Ping.pkl')

# Title of the app
# st.title("Text Classification Prediction")

# Input box for user to enter JSON
json_input = st.text_area("Enter JSON input:")

# Predict and display the result when button is clicked
if st.button("Predict"):
    if json_input:
        try:
            # Parse the JSON input
            data = json.loads(json_input)
            
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            
            # Make prediction
            predictions = model.predict(df)
            predict_proba = model.predict_proba(df)
            
            # Simplify the predict_proba output and get the probability score of class 1
            simplified_proba = [proba[pred] for pred, proba in zip(predictions, predict_proba)]
            probability_score_of_1 = [proba[1] for proba in predict_proba]
            
            # Transform probability score to 0 or 1 based on the threshold 0.5
            transformed_proba = [1 if score >= 0.5 else 0 for score in probability_score_of_1]

            # Prepare the response
            response = {
                # 'predictions': predictions.tolist(),
                # 'simplified_proba': simplified_proba,
                # 'probability_score_of_1': probability_score_of_1,
                # 'transformed_proba': transformed_proba,
                'probability_score': predict_proba[:, 1].tolist(),
            }
            
            # Display results
            st.json(response)
        except json.JSONDecodeError:
            st.write("Invalid JSON input. Please enter valid JSON.")
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter some JSON input.")