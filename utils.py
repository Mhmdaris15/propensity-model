import ipaddress

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