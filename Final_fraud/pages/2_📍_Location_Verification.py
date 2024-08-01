import streamlit as st
import requests
from requests.auth import HTTPBasicAuth

st.title("Location Verification")

def get_real_time_location():
    ipinfo_token = "d0aa0106d602f4"
    url = f'https://ipinfo.io/json?token={ipinfo_token}'
    response = requests.get(url)
    response.raise_for_status()
    location_data = response.json()
    lat, lng = map(float, location_data['loc'].split(','))
    return lat, lng

def geocode_location(city_name):
    url = f'https://nominatim.openstreetmap.org/search?q={city_name}&format=json'
    headers = {'User-Agent': 'YourAppName/1.0 (your-email@example.com)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    geocode_data = response.json()
    location = geocode_data[0]
    lat, lon = float(location['lat']), float(location['lon'])
    return lat, lon

def verify_location(real_time_location, user_location):
    tolerance = 0.05
    return (abs(real_time_location[0] - user_location[0]) < tolerance and 
            abs(real_time_location[1] - user_location[1]) < tolerance)

def is_fraudulent_company(company_name, fraudulent_companies):
    return company_name.lower() in (name.lower() for name in fraudulent_companies)

def send_sms_twilio(account_sid, auth_token, to_phone_number, from_phone_number, message):
    url = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json'
    data = {'To': to_phone_number, 'From': from_phone_number, 'Body': message}
    response = requests.post(url, data=data, auth=HTTPBasicAuth(account_sid, auth_token))
    if response.status_code == 201:
        st.success("Message sent successfully.")
    else:
        st.error(f"Failed to send message: {response.status_code} - {response.text}")

# Main logic for location verification
try:
    real_time_location = get_real_time_location()
    #st.write(f"Real-Time Location: {real_time_location}")

    user_city = st.text_input("Enter your city (e.g., Kolkata, Mumbai): ")
    if user_city:
        user_location = geocode_location(user_city)
        #st.write(f"User Input Location (Geocoded): {user_location}")

        fraudulent_companies = [
            'Enron',
            'Bernard L. Madoff Investment Securities LLC',
            'WorldCom',
            'Theranos',
            'Lehman Brothers',
            'Wirecard',
            'Luckin Coffee',
            'Mahindra Satyam',
            'HealthSouth',
            'Wells Fargo'
        ]
        
        company_name = st.text_input("Enter the company name for the transaction: ")
        if company_name:
            location_mismatch = not verify_location(real_time_location, user_location)
            fraudulent_company = is_fraudulent_company(company_name, fraudulent_companies)

            message = ""
            if location_mismatch:
                st.write("Fraud detected: Location mismatch.")
                message += "Potential fraud detected: Location mismatch. "
            if fraudulent_company:
                st.write(f"Fraud detected: Transaction with fraudulent company ({company_name}).")
                message += f"Potential fraud detected: Transaction with fraudulent company ({company_name})."

            if message:
                account_sid = "AC32a4144860f2171bc04a8b8321e2682f"
                auth_token = "25df4623340233a4bd0c8a07b99b47b0"
                to_phone_number = "+919875622707"
                from_phone_number = "+19382226379"
                
                send_sms_twilio(account_sid, auth_token, to_phone_number, from_phone_number, message)
            else:
                st.write("Transaction seems safe.")
except Exception as e:
    st.error(e)
