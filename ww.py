import requests
import json
from datetime import datetime, timedelta
import time

# Your API key and base URL
API_KEY = '8a1046e19a684b619aa133633240709'
BASE_URL = 'https://api.weatherapi.com/v1/history.json'

# List of cities (replace with actual city names)
cities = ['Abohar', 'Adilabad', 'Agartala', 'Agra', 'Ahmadnagar', 'Ahmedabad', 'Aizawl  ', 'Ajmer', 'Akola', 'Alappuzha', 'Aligarh', 'Alipurduar', 'Allahabad', 'Alwar', 'Ambala', 'Amaravati', 'Amritsar', 'Asansol', 'Aurangabad', 'Aurangabad', 'Bakshpur', 'Bamanpuri', 'Baramula', 'Barddhaman', 'Bareilly', 'Belgaum', 'Bellary', 'Bengaluru', 'Bhagalpur', 'Bharatpur', 'Bharauri', 'Bhatpara', 'Bhavnagar', 'Bhilai', 'Bhilwara', 'Bhiwandi', 'Bhiwani', 'Bhopal ', 'Bhubaneshwar', 'Bhuj', 'Bhusaval', 'Bidar', 'Bijapur', 'Bikaner', 'Bilaspur', 'Brahmapur', 'Budaun', 'Bulandshahr', 'Calicut', 'Chanda', 'Chandigarh ', 'Chennai', 'Chikka Mandya', 'Chirala', 'Coimbatore', 'Cuddalore', 'Cuttack', 'Daman', 'Davangere', 'DehraDun', 'Delhi', 'Dhanbad', 'Dibrugarh', 'Dindigul', 'Dispur', 'Diu', 'Faridabad', 'Firozabad', 'Fyzabad', 'Gangtok', 'Gaya', 'Ghandinagar', 'Ghaziabad', 'Gopalpur', 'Gulbarga', 'Guntur', 'Gurugram', 'Guwahati', 'Gwalior', 'Haldia', 'Haora', 'Hapur', 'Haripur', 'Hata', 'Hindupur', 'Hisar', 'Hospet', 'Hubli', 'Hyderabad', 'Imphal', 'Indore', 'Itanagar', 'Jabalpur', 'Jaipur', 'Jammu', 'Jamshedpur', 'Jhansi', 'Jodhpur', 'Jorhat', 'Kagaznagar', 'Kakinada', 'Kalyan', 'Karimnagar', 'Karnal', 'Karur', 'Kavaratti', 'Khammam', 'Khanapur', 'Kochi', 'Kohima', 'Kolar', 'Kolhapur', 'Kolkata ', 'Kollam', 'Kota', 'Krishnanagar', 'Krishnapuram', 'Kumbakonam', 'Kurnool', 'Latur', 'Lucknow', 'Ludhiana', 'Machilipatnam', 'Madurai', 'Mahabubnagar', 'Malegaon Camp', 'Mangalore', 'Mathura', 'Meerut', 'Mirzapur', 'Moradabad', 'Mumbai', 'Muzaffarnagar', 'Muzaffarpur', 'Mysore', 'Nagercoil', 'Nalgonda', 'Nanded', 'Nandyal', 'Nasik', 'Navsari', 'Nellore', 'New Delhi', 'Nizamabad', 'Ongole', 'Pali', 'Panaji', 'Panchkula', 'Panipat', 'Parbhani', 'Pathankot', 'Patiala', 'Patna', 'Pilibhit', 'Porbandar', 'Port Blair', 'Proddatur', 'Puducherry', 'Pune', 'Puri', 'Purnea', 'Raichur', 'Raipur', 'Rajahmundry', 'Rajapalaiyam', 'Rajkot', 'Ramagundam', 'Rampura', 'Ranchi', 'Ratlam', 'Raurkela', 'Rohtak', 'Saharanpur', 'Saidapur', 'Saidpur', 'Salem', 'Samlaipadar', 'Sangli', 'Saugor', 'Shahbazpur', 'Shiliguri', 'Shillong ', 'Shimla', 'Shimoga', 'Sikar', 'Silchar', 'Silvassa', 'Sirsa', 'Sonipat', 'Srinagar', 'Surat', 'Tezpur', 'Thanjavur', 'Tharati Etawah', 'Thiruvananthapuram', 'Tiruchchirappalli', 'Tirunelveli', 'Tirupati', 'Tiruvannamalai', 'Tonk', 'Tuticorin', 'Udaipur', 'Ujjain', 'Vadodara', 'Valparai', 'Varanasi', 'Vellore', 'Vishakhapatnam', 'Vizianagaram', 'Warangal', 'Jorapokhar', 'Brajrajnagar', 'Talcher']

# Function to get weather data for a specific city and date
def get_weather_data(city, date):
    url = f'{BASE_URL}?key={API_KEY}&q={city}&dt={date}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error fetching data for {city} on {date}: {response.status_code}')
        print('Response content:', response.text)  # Print response content for debugging
        return None


# Function to get weather data for all cities over the past 6 months
def get_past_weather_data(cities):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Past 6 months

    all_weather_data = {}

    current_date = start_date
    while current_date <= end_date:
        formatted_date = current_date.strftime('%Y-%m-%d')
        print(f'Fetching data for {formatted_date}')

        for city in cities:
            data = get_weather_data(city, formatted_date)
            if data:
                if city not in all_weather_data:
                    all_weather_data[city] = {}
                all_weather_data[city][formatted_date] = data

        current_date += timedelta(days=1)
        time.sleep(1)  # Be respectful to API rate limits

    return all_weather_data

# Fetch and save weather data
weather_data = get_past_weather_data(cities)
with open('weather_data.json', 'w') as f:
    json.dump(weather_data, f, indent=4)

print('Weather data retrieval complete. Saved to weather_data.json')
