import openmeteo_requests
import requests
from tkinter import Tk, Label, Entry, Button, messagebox
from openmeteo_sdk.Variable import Variable


API_KEY = "R3IMUd2rYkCKeS4Q"


def get_location(city):
    base_url = f"https://www.meteoblue.com/en/server/search/query3?query={city}&apikey={API_KEY}"
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        if data['count'] == 0:
            return None, "City not found."
        # Take the first result (most relevant city)
        location = data['results'][0]
        lat = location['lat']
        lon = location['lon']

        return lat, lon, None
    except requests.exceptions.RequestException as e:
        return None, None, str(e)


def get_weather_data(city):
    """
    Fetches weather data for a given city using the Open-Meteo API.
    """
    # Step 1: Get the location (lat, lon) using the Meteoblue Location Search API
    lat, lon, error_message = get_location(city)
    if not lat or not lon:
        return None, error_message

    # Step 2: Fetch weather data from the Open-Meteo API
    om = openmeteo_requests.Client()

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m"]
    }

    try:
        responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]

        # Get current weather information
        current = response.Current()
        current_variables = list(map(lambda i: current.Variables(i), range(0, current.VariablesLength())))

        # Get the temperature and humidity values
        current_temperature_2m = next(
            filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, current_variables))
        current_relative_humidity_2m = next(
            filter(lambda x: x.Variable() == Variable.relative_humidity and x.Altitude() == 2, current_variables))

        return {
                   "city": city,
                   "temperature": current_temperature_2m.Value(),
                   "humidity": current_relative_humidity_2m.Value()
               }, None
    except Exception as e:
        return None, str(e)


def determine_weather_condition(temperature, humidity):
    """
    Determines weather condition based on temperature and humidity.
    """
    if temperature > 40:
        return "Heatwave"
    elif temperature < -10:
        return "Cold snap"
    elif humidity > 90:
        return "Fog"
    elif temperature > 30:
        return "Clear sky"
    elif 20 < temperature <= 30:
        return "Partly cloudy"
    elif 10 < temperature <= 20:
        return "Cloudy"
    elif 0 <= temperature <= 10:
        return "Overcast"
    elif temperature < 0:
        return "Snow"
    elif humidity > 80:
        return "Rain"
    elif humidity < 30:
        return "Windy"
    else:
        return "Clear sky"


def display_weather(data, window):
    """
    Displays the weather information in the GUI window.
    """
    if data:
        city = data.get("city", "Unknown")
        temp = data.get("temperature", "N/A")
        humidity = data.get("humidity", "N/A")

        # Determine weather condition based on temperature and humidity
        condition = determine_weather_condition(temp, humidity)

        weather_info = (
            f"Weather for {city}:\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Weather: {condition}"
        )
        result_label.config(text=weather_info)
    else:
        messagebox.showerror("Error", "No weather data available.")


def get_and_display_weather():
    """
    Handles the event of clicking the 'Get Weather' button.
    """
    city = city_entry.get()
    if not city:
        messagebox.showwarning("Input Error", "Please enter a city name.")
        return

    # Step 1: Get the weather data from Open-Meteo API
    data, error_message = get_weather_data(city)
    if data:
        display_weather(data, window)
    else:
        messagebox.showerror("Error", f"Error: {error_message}")


# Create the main window
window = Tk()
window.title("Bolt of Zeus")

# Set the window size
window.geometry("500x350")  # Width x Height

# Create input field and button for user interaction
city_label = Label(window, text="Enter city name:")
city_label.pack(pady=5)

city_entry = Entry(window)
city_entry.pack(pady=5)

get_weather_button = Button(window, text="Get Weather", command=get_and_display_weather)
get_weather_button.pack(pady=10)

# Label to display weather results
result_label = Label(window, text="", justify="left")
result_label.pack(pady=10)

# Start the GUI event loop
window.mainloop()
