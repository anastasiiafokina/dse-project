# --- Data Loading and Processing Functions ---
import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the city data from a CSV file."""
    return pd.read_csv(filepath)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth using the Haversine formula."""
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(
        np.sqrt(a), np.sqrt(1 - a)
    )  # angular distance between two points, expressed in radians
    return R * c  # Distance in kilometers

def calculate_distance_between_cities(city1_name, city2_name, data):
    """Calculate the distance between two cities by their names."""
    city1 = data[data["City"] == city1_name].iloc[0]
    city2 = data[data["City"] == city2_name].iloc[0]

    # Getting coordinates of cities
    lat1, lon1 = city1["Latitude"], city1["Longitude"]
    lat2, lon2 = city2["Latitude"], city2["Longitude"]

    # Calculate the distance using the haversine function
    distance = haversine(lat1, lon1, lat2, lon2)

    print(f"Distance between {city1_name} and {city2_name}: {distance:.2f} km")
    return distance

def find_nearest_cities(data, current_city, num_nearest=3):
    """Find the 3 nearest cities to the current city, based on latitude and longitude."""
    distances = []
    for _, city in data.iterrows():
        if city["City"] != current_city["City"]:  # Skip the same city
            dist = haversine(
                current_city["Latitude"],
                current_city["Longitude"],
                city["Latitude"],
                city["Longitude"],
            )
            distances.append((city, dist))

    # Sorting the list of distances and selecting the nearest cities
    distances.sort(key=lambda x: x[1])
    nearest_cities = distances[:num_nearest]

    # Display nearest cities and distances
    print(f"\nThe nearest {num_nearest} cities to {current_city['City']}:")
    for city, dist in nearest_cities:
        print(f"-{city['City']} is {dist:.2f} km away")

    # Add rank to each nearest city (0 for nearest, 1 for second nearest, etc.)
    return [
        {
            "City": city["City"],
            "Country": city["Country"],
            "Population": city["Population"],
            "Latitude": city["Latitude"],
            "Longitude": city["Longitude"],
            "Rank": rank,
        }
        for rank, (city, _) in enumerate(nearest_cities)
    ]