import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
# input file
model = joblib.load('model/Supermodel API Pre-Ping.pkl')
from datetime import datetime, timedelta
from utils import numerize_ip, get_time_category

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
        "AS Description": "ATT-INTERNET4",
        "country": "US",
        "state": "OR",
        "city": "GRANTS PASS",
        "postalcode": 97526,
        "connection_type": "Cable/DSL",
        "coreg_path": "15",
        "isp": "AT&T Wireless",
        "Male/Female": 0,
        "source": "whatifmedia-linkout",
        "subid": "1006",
        "Age": 61,
        "Latitude (generated)": 42.4394,
        "Longitude (generated)": -123.3272,
        "IP Address": "107.116.110.60"
    },
    {
        "Timestamp": "2024-05-18 00:00:00",
        "AS Description": "CELLCO-PART",
        "country": "US",
        "state": "NC",
        "city": "LEXINGTON",
        "postalcode": 27292,
        "connection_type": "Cellular",
        "coreg_path": "5",
        "isp": "Verizon Wireless",
        "Male/Female": 0,
        "source": "whatifmedia-linkout",
        "subid": "1799",
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
      "AS Description": "ATT-INTERNET4",
      "country": "US",
      "state": "FL",
      "city": "FORT LAUDERDALE",
      "postalcode": 33325,
      "connection_type": "Cable/DSL",
      "coreg_path": "3",
      "isp": "AT&T Internet",
      "Male/Female": 0,
      "source": "whatifmedia-linkout",
      "subid": "2027",
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
- **AS Description**: String
- **country**: String
- **state**: String
- **city**: String
- **postalcode**: Integer
- **connection_type**: String
- **coreg_path**: String
- **isp**: String
- **Male/Female**: Integer (0 for Male, 1 for Female)
- **source**: String
- **subid**: String
- **Age**: Integer
- **Latitude (generated)**: Float
- **Longitude (generated)**: Float
- **IP Address**: String

## Notes

- The `Timestamp` field is used to extract the hour and derive the `Time Category`.
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
      "AS Description": "ATT-INTERNET4",
      "country": "US",
      "state": "OR",
      "city": "GRANTS PASS",
      "postalcode": 97526,
      "connection_type": "Cable/DSL",
      "coreg_path": "15",
      "isp": "AT&T Wireless",
      "Male/Female": 0,
      "source": "whatifmedia-linkout",
      "subid": "1006",
      "Age": 61,
      "Latitude (generated)": 42.4394,
      "Longitude (generated)": -123.3272,
      "IP Address": "107.116.110.60"
  },
  {
      "Timestamp": "2024-05-18 00:00:00",
      "AS Description": "CELLCO-PART",
      "country": "US",
      "state": "NC",
      "city": "LEXINGTON",
      "postalcode": 27292,
      "connection_type": "Cellular",
      "coreg_path": "5",
      "isp": "Verizon Wireless",
      "Male/Female": 0,
      "source": "whatifmedia-linkout",
      "subid": "1799",
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

            # Initialize a list to store input data for all records
            all_input_data = []

            # Process each record in the data list
            for data in data_list:
                # Parse input data
                timestamp = data['Timestamp']
                as_description = data['AS Description']
                country = data['country']
                state = data['state']
                city = data['city']
                postalcode = data['postalcode']
                connection_type = data['connection_type']
                coreg_path = data['coreg_path']
                isp = data['isp']
                male_female = data['Male/Female']
                source = data['source']
                subid = data['subid']
                age = data['Age']
                latitude = data['Latitude (generated)']
                longitude = data['Longitude (generated)']
                ip_address = data['IP Address']

                # Compute derived fields
                state_city = f"{state} - {city}"
                source_subid = f"{source} - {subid}"
                hour = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').hour
                time_category = get_time_category(hour)
                ip_address_numerized = numerize_ip(ip_address)

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
                    "Male/Female": male_female,
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

                # Append the input data to the list
                all_input_data.append(input_data)

            
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(all_input_data)
            
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
            # st.json(response)
            st.json(predict_proba[:, 1].tolist())
        except json.JSONDecodeError:
            st.write("Invalid JSON input. Please enter valid JSON.")
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter some JSON input.")


json_input_2_sample = """{
    "Timestamp": "2024-05-13 00:00:00",
    "AS Description": "ATT-INTERNET4",
    "country": "US",
    "state": "FL",
    "city": "FORT LAUDERDALE",
    "postalcode": 33325,
    "connection_type": "Cable/DSL",
    "coreg_path": "3",
    "isp": "AT&T Internet",
    "Male/Female": 0,
    "source": "whatifmedia-linkout",
    "subid": "2027",
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
            as_description = data['AS Description']
            country = data['country']
            state = data['state']
            city = data['city']
            postalcode = data['postalcode']
            connection_type = data['connection_type']
            coreg_path = data['coreg_path']
            isp = data['isp']
            male_female = data['Male/Female']
            source = data['source']
            subid = data['subid']
            age = data['Age']
            latitude = data['Latitude (generated)']
            longitude = data['Longitude (generated)']
            ip_address = data['IP Address']

            # Compute derived fields
            state_city = f"{state} - {city}"
            source_subid = f"{source} - {subid}"
            hour = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').hour
            time_category = get_time_category(hour)
            ip_address_numerized = numerize_ip(ip_address)

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
                "Male/Female": male_female,
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
            
            # Convert the JSON data to a DataFrame
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