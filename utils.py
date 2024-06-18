import ipaddress
import pandas as pd
from datetime import datetime, timedelta

# Function to numerize IP address
def numerize_ip(ip_address: str):
    return int(ipaddress.ip_address(ip_address))

# Function to determine Time Category based on Hour
def get_time_category(hour: int):
    if 0 <= hour < 6:
        return "0 - 6"
    elif 6 <= hour < 12:
        return "6 - 12"
    elif 12 <= hour < 18:
        return "12 - 18"
    else:
        return "18 - 24"

def get_feature_engineerings(timestamps: datetime) :
    dow = timestamps.dayofweek
    dow_dict = {
        0 : 'Monday',
        1 : 'Tuesday',
        2 : 'Wednesday',
        3 : 'Thursday',
        4 : 'Friday',
        5 : 'Saturday',
        6 : 'Sunday'
    }
    dow = dow_dict[dow]

    timestamps = timestamps.to_period(freq = 's')
    hour = timestamps.hour
    day = timestamps.day
    week = timestamps.week

    return [hour, day, week, dow]

def get_month_dobs(dob: datetime) :
    dob = dob.to_period(freq = 'D')
    month = dob.month.astype(int)
    return month.values

def get_feature_engineering(timestamp: str) :
    timestamp = pd.to_datetime([timestamp], format = '%Y-%m-%d %H:%M:%S').to_period(freq = 's')
    hour = int(timestamp.hour[0])
    day = int(timestamp.day[0])
    week = int(timestamp.week[0])
    dow = str(timestamp.dayofweek.map({
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    })[0])

    return hour, day, week, dow

def get_month_dob(dob: str) :
    dob = pd.to_datetime([dob], format = '%Y-%m-%d').to_period('D')
    month = int(dob.month[0])
    return month