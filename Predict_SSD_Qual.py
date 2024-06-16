import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import ipaddress

# Description
st.markdown("""
## Propensity Model Demonstration

This application is designed to predict whether an `SSD` is qualified or not based on various features. The model used in this application is built with a VotingClassifier combining CatBoost and Logistic Regression. By leveraging the strengths of both classifiers, the model aims to provide accurate predictions and high confidence scores.

### Features

The model considers the following features to make predictions:

- **AS Description**: The autonomous system description of the network.
- **Country**: The country where the request originated.
- **State**: The state where the request originated.
- **City**: The city where the request originated.
- **Postal Code**: The postal code of the location.
- **Connection Type**: The type of internet connection (e.g., Cable/DSL, Corporate, Cellular, Satellite).
- **Coreg Path**: The coregistration path identifier.
- **ISP**: The Internet Service Provider.
- **Male/Female**: The gender of the user (1 for Male, 0 for Female).
- **Source**: The source of the request.
- **Sub ID**: The sub-identifier associated with the source.
- **Age**: The age of the user. It's derived from difference betweey current year (2024) and year of `dob` (Date of Birthdate)
- **Latitude (generated)**: The generated latitude of the location.
- **Longitude (generated)**: The generated longitude of the location.
- **Hour**: The hour of the day when the request was made. It's derived from `Datetime` column, where we extracted the hour of the day.
- **IP Address**: The IP address of the request, which is then numerized for the model.

### Derived Features

Some features are derived from the input data to enhance the model's predictive power:

- **State + City**: A combined feature of the state and city.
- **Source + Sub Id**: A combined feature of the source and sub-identifier.
- **Time Category**: Categorized time of the day based on the hour (e.g., "0 - 6", "6 - 12", "12 - 18", "18 - 24").
- **IP Address Numerized**: The IP address converted to a numerical format for better processing by the model.

### Feature Engineering

The following feature engineering steps are implemented in the application:

1. **IP Address Numerization**: The IP address is converted into a numerical format using Python's `ipaddress` module. This helps in transforming categorical IP data into a format that can be easily processed by machine learning algorithms.
2. **Combining Features**: New features like "State + City" and "Source + Sub Id" are created by combining existing features. This provides additional context to the model and can improve predictive performance.
3. **Time Categorization**: The hour of the day is categorized into distinct time ranges. This helps the model to capture patterns that occur during different times of the day.
""")

# Define available pages
pages = {
    "Home": "home",
    "Page 1": "page1",
}

# Load the pre-trained model
model = joblib.load('model/Supermodel API Pre-Ping.pkl')

# Function to numerize IP address
def numerize_ip(ip_address):
    return int(ipaddress.ip_address(ip_address))

# Function to determine Time Category based on Hour
def get_time_category(hour):
    if 0 <= hour < 6:
        return "0 - 6"
    elif 6 <= hour < 12:
        return "6 - 12"
    elif 12 <= hour < 18:
        return "12 - 18"
    else:
        return "18 - 24"

# Title of the app
st.title("Structured Data Prediction")

# Cache the data loading function
@st.cache_data
def load_data():
    df =  pd.read_excel('data/Processed Linkout ML Propensity Data V8.xlsx')
    as_description_list = df['AS Description'].unique().tolist()
    country_list = df['country'].unique().tolist()
    state_list = df['state'].unique().tolist()
    city_list = df['city'].unique().tolist()
    isp_list = df['isp'].unique().tolist()
    source_list = df['source'].unique().tolist()
    subid_list = df['subid'].unique().tolist()
    connection_type_list = df['connection_type'].unique().tolist()
    coreg_path_list = df['coreg_path'].unique().tolist()
    return df, as_description_list, country_list, state_list, city_list, isp_list, source_list, subid_list, connection_type_list, coreg_path_list

# Load the data
df, as_description_list, country_list, state_list, city_list, isp_list, source_list, subid_list, connection_type_list, coreg_path_list = load_data()

# Form for input
with st.form(key='input_form'):
    as_description = st.selectbox("AS Description", options=as_description_list)
    country = st.selectbox("Country", options=country_list)
    state = st.selectbox("State", options=state_list)
    city = st.selectbox("City", options=city_list)
    postalcode = st.number_input("Postal Code", value=33603)
    connection_type = st.selectbox("Connection Type", options=connection_type_list)
    coreg_path = st.selectbox("Coreg Path", options=coreg_path_list)
    isp = st.selectbox("ISP", options=isp_list)
    male_female = st.radio("Gender", options=["Male", "Female"], index=0)
    source = st.selectbox("Source", options=source_list)
    subid = st.selectbox("Sub ID", options=subid_list)
    age = st.number_input("Age", value=62)
    latitude = st.number_input("Latitude (generated)", value=27.9478)
    longitude = st.number_input("Longitude (generated)", value=-82.4584)
    hour = st.number_input("Hour", value=22, min_value=0, max_value=23)
    ip_address = st.text_input("IP Address", value="35.138.16.9")

    # Derived fields (read-only)
    state_city = f"{state} - {city}"
    source_subid = f"{source} - {subid}"
    time_category = get_time_category(hour)
    ip_address_numerized = numerize_ip(ip_address)

    st.write(f"State + City: {state_city}")
    st.write(f"Source + Sub Id: {source_subid}")
    st.write(f"Time Category: {time_category}")
    st.write(f"IP Address Numerized: {ip_address_numerized}")

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# When the form is submitted
if submit_button:
    # Create a dictionary with the input data
    input_data = {
        "AS Description": as_description,
        "country": country,
        "state": state,
        "city": city,
        "postalcode": postalcode,
        "connection_type": connection_type,
        "coreg_path": coreg_path,
        "isp": isp,
        "Male/Female": 1 if male_female == "Male" else 0,
        "source": source,
        "subid": subid,
        "Age": age,
        "Latitude (generated)": latitude,
        "Longitude (generated)": longitude,
        "State + City": state_city,
        "Source + Sub Id": source_subid,
        "Hour": hour,
        "Time Category": time_category,
        "IP Address Numerized": ip_address_numerized
    }

    # Convert the input data to a DataFrame
    df = pd.DataFrame([input_data])

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
        'predictions': predictions.tolist(),
        'simplified_proba': simplified_proba,
        'probability_score_of_1': probability_score_of_1,
        'transformed_proba': transformed_proba
    }

    final_response = predict_proba[:, 1].tolist()
    
    # Display final results
    st.write("## Probability Score:")
    st.write(f"### The probability score of the SSD being qualified is: {final_response[0]}")
    
