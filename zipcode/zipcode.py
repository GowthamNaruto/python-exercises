# PIP install geopy
from geopy.geocoders import Nominatim

# Create the geolocater object with a user-agent
geolocator = Nominatim(user_agent="geoapiExercises")

# Get the city name from the user
place = input("Enter city name: ")

# Geocode the location
location = geolocator.geocode(place)

# Print the  geolocation details
print(location)
