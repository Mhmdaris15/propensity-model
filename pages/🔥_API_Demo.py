import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
# input file
model = joblib.load('model/Supermodel API Pre-Ping.sklearn')
imputer = joblib.load('model/SimpleImputer.sklearn')
from datetime import datetime, timedelta
from utils import numerize_ip, get_time_category, get_feature_engineerings, get_month_dobs, get_feature_engineering, get_month_dob

# Title of the app
# st.title("Text Classification Prediction")

st.markdown("""
# API Documentation of Propensity Model

## Endpoints

### 1. **POST /predicts**

Predict the probability for multiple data entries.

- **URL**: `/predicts`
- **Method**: `POST`
- **Request Body**: 
  - A JSON array containing multiple data objects.
  
  #### Example Request:
  ```json
  [
    {
        "Timestamp": "2024-05-01 00:00:00",
        "DOB" : "1985-03-21",
        "AS Description": "ATT-INTERNET4",
        "Country Code": "US",
        "State": "OR",
        "City": "GRANTS PASS",
        "Postalcode": 97526,
        "Coreg Path": "15",
        "Male/Female": 0,
        "Source": "whatifmedia-linkout",
        "Subid": "1006",
        "Age": 61,
        "Latitude (generated)": 42.4394,
        "Longitude (generated)": -123.3272,
        "IP Address": "107.116.110.60"
    },
    {
        "Timestamp": "2024-05-18 00:00:00",
        "DOB" : "1983-06-14",
        "AS Description": "CELLCO-PART",
        "Country Code": "US",
        "State": "NC",
        "City": "LEXINGTON",
        "Postalcode": 27292,
        "Coreg Path": "5",
        "Male/Female": 0,
        "Source": "whatifmedia-linkout",
        "Subid": "1799",
        "Age": 53,
        "Latitude (generated)": 35.824,
        "Longitude (generated)": -80.2534,
        "IP Address": "174.210.72.204"
    }
  ]
  ```

- **Response**: 
  - A JSON array containing the prediction probabilities for each input data object.
  
  #### Example Response:
  ```json
  [
    0.5085516174353844,
    0.3242479008363784
  ]
  ```

### 2. **POST /predict**

Predict the probability for a single data entry.

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**: 
  - A single JSON object containing the data.
  
  #### Example Request:
  ```json
  {
    "Timestamp": "2024-05-13 00:00:00",
    "DOB" : "1983-06-14",
    "AS Description": "ATT-INTERNET4",
    "Country Code": "US",
    "State": "FL",
    "City": "FORT LAUDERDALE",
    "Postalcode": 33325,
    "Coreg Path": "3",
    "Male/Female": 0,
    "Source": "whatifmedia-linkout",
    "Subid": "2027",
    "Age": 62,
    "Latitude (generated)": 26.1223,
    "Longitude (generated)": -80.1434,
    "IP Address": "99.73.70.54"
  }
  ```

- **Response**: 
  - A string of float containing the prediction probability.
  
  #### Example Response:
  ```json
    0.3579423856752602
  ```

## Data Fields

Each input object should contain the following fields:

- **Timestamp**: String (format `YYYY-MM-DD HH:MM:SS`)
- **DOB** : String (format = `YYYY-MM-DD`)
- **AS Description**: String
- **Country**: String
- **State**: String
- **City**: String
- **Postalcode**: Integer
- **Coreg Path**: String
- **Male/Female**: Integer (0 for Male, 1 for Female)
- **Source**: String
- **Subid**: String
- **Age**: Integer
- **Latitude (generated)**: Float
- **Longitude (generated)**: Float
- **IP Address**: String

## Notes

- The `Timestamp` field is used to extract the hour, day, week, and DOW, and to derive the `Time Category`.
- The `IP Address` is numerized to create `IP Address Numerized`.
- Additional fields like `State + City`, `Source + Sub Id`, `Hour`, and `Time Category` are derived from the provided fields.

## Error Handling

- Ensure that all required fields are provided in the correct format.
- The response will include error messages if any required fields are missing or incorrectly formatted.
""")

st.write("## API Demonstration")

json_input_1_sample = """[
  {
      "Timestamp": "2024-05-01 00:00:00",
      "DOB" : "1965-03-08",
      "AS Description": "ATT-INTERNET4",
      "Country Code": "US",
      "State": "OR",
      "City": "GRANTS PASS",
      "Postalcode": 97526,
      "Coreg Path": "15",
      "Male/Female": 0,
      "Source": "whatifmedia-linkout",
      "Subid": "1006",
      "Age": 61,
      "Latitude (generated)": 42.4394,
      "Longitude (generated)": -123.3272,
      "IP Address": "107.116.110.60"
  },
  {
      "Timestamp": "2024-05-18 00:00:00",
      "DOB" : "1965-12-15",
      "AS Description": "CELLCO-PART",
      "Country Code": "US",
      "State": "NC",
      "City": "LEXINGTON",
      "Postalcode": 27292,
      "Coreg Path": "5",
      "Male/Female": 0,
      "Source": "whatifmedia-linkout",
      "Subid": "1799",
      "Age": 53,
      "Latitude (generated)": 35.824,
      "Longitude (generated)": -80.2534,
      "IP Address": "174.210.72.204"
  }
]"""

# Input box for user to enter JSON
json_input_1 = st.text_area("Enter an array of json objects for multiple data entries (/predicts):", height=400, value=json_input_1_sample)

# Predict and display the result when button is clicked
if st.button("Predict", key="predicts"):
    if json_input_1:
        try:
            # Parse the JSON input
            data_list = json.loads(json_input_1)
            df = pd.DataFrame.from_records(data_list)

            df['State + City'] = df['State'] + ' - ' + df['City']
            df['Source + Sub Id'] = df['Source'] + ' - ' + df['Subid']
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df[['Hour', 'Day', 'Week', 'DOW']] = df['Timestamp'].apply(get_feature_engineerings).apply(pd.Series)
            df['Month_Dob'] = df['DOB'].apply(get_month_dob)
            df['Time Category'] = df['Hour'].apply(get_time_category)
            df['IP Numerized'] = df['IP Address'].apply(numerize_ip)

            df = df[[
                'AS Description', 'Country Code', 'State', 'City', 'Postalcode', 'Coreg Path',
                'Male/Female', 'Source', 'Subid', 'Age', 'IP Numerized', 'Latitude (generated)',
                'Longitude (generated)', 'Source + Sub Id', 'State + City', 'Hour', 'Day', 'Week',
                'DOW', 'Time Category', 'Month_Dob'
            ]]
            st.write(df)
            
            # Make prediction
            df[df.columns] = imputer.transform(df)

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
            # st.json(response)
            st.json(predict_proba[:, 1].tolist())
        except json.JSONDecodeError:
            st.write("Invalid JSON input. Please enter valid JSON.")
        # except Exception as e:
        #     st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter some JSON input.")


json_input_2_sample = """{
    "Timestamp": "2024-05-13 00:00:00",
    "DOB" : "1975-03-22",
    "AS Description": "ATT-INTERNET4",
    "Country Code": "US",
    "State": "FL",
    "City": "FORT LAUDERDALE",
    "Postalcode": 33325,
    "Coreg Path": "3",
    "Male/Female": 0,
    "Source": "whatifmedia-linkout",
    "Subid": "2027",
    "Age": 62,
    "Latitude (generated)": 26.1223,
    "Longitude (generated)": -80.1434,
    "IP Address": "99.73.70.54"
}"""

# Input box for user to enter JSON
json_input_2 = st.text_area("Enter JSON input for a single data entry object (/predict):", height=350, value=json_input_2_sample)

# Predict and display the result when button is clicked
if st.button("Predict", key="predict"):
    if json_input_2:
        try:
            # Parse the JSON input
            data = json.loads(json_input_2)

            # Parse input data
            timestamp = data['Timestamp']
            dob = data['DOB']
            as_description = data['AS Description']
            country = data['Country Code']
            state = data['State']
            city = data['City']
            postalcode = data['Postalcode']
            coreg_path = data['Coreg Path']
            male_female = data['Male/Female']
            source = data['Source']
            subid = data['Subid']
            age = data['Age']
            latitude = data['Latitude (generated)']
            longitude = data['Longitude (generated)']
            ip_address = data['IP Address']

            # Compute derived fields
            state_city = f"{state} - {city}"
            source_subid = f"{source} - {subid}"
            hour, day, week, dow = get_feature_engineering(timestamp)
            month_dob = get_month_dob(dob)
            time_category = get_time_category(hour)
            ip_address_numerized = numerize_ip(ip_address)

            # Create a dictionary with the input data
            input_data = {
                "AS Description": as_description,
                "Country Code": country,
                "State": state,
                "City": city,
                "Postalcode": postalcode,
                "Coreg Path": coreg_path,
                "Male/Female": male_female,
                "Source": source,
                "Subid": subid,
                "Age": age,
                "IP Numerized": ip_address_numerized,
                "Latitude (generated)": latitude,
                "Longitude (generated)": longitude,
                "Source + Sub Id": source_subid,
                "State + City": state_city,
                "Hour": hour,
                "Day" : day,
                "Week" : week,
                "DOW" : dow,
                "Time Category": time_category,
                "Month_Dob" : month_dob
            }
            
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame([input_data])
            
            # Make prediction
            df[df.columns] = imputer.transform(df)
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
            st.write(predict_proba[:, 1].tolist()[0])
        except json.JSONDecodeError:
            st.write("Invalid JSON input. Please enter valid JSON.")
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter some JSON input.")