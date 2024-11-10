import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

#%%
# --- Data Loading and Processing Functions ---

def load_data(filepath):
    """Load the city data from a CSV file."""
    return pd.read_csv(filepath)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth using the Haversine formula."""
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # Distance in kilometers

def find_nearest_cities(data, current_city, num_nearest=3):
    """Find the nearest cities to the current city, based on latitude and longitude."""
    distances = []
    for _, city in data.iterrows():
        if city['city'] != current_city['city']:  # Skip the same city
            dist = haversine(current_city['latitude'], current_city['longitude'], city['latitude'], city['longitude'])
            distances.append((city, dist))
    
    # Sort by distance and get the top nearest cities
    nearest_cities = sorted(distances, key=lambda x: x[1])[:num_nearest]
    # Add rank to each nearest city (0 for nearest, 1 for second nearest, etc.)
    return [{'city': city['city'], 'country': city['country'], 'population': city['population'],
             'latitude': city['latitude'], 'longitude': city['longitude'], 'rank': rank}
            for rank, (city, _) in enumerate(nearest_cities)]
    
#%%
# --- Travel Logic Functions ---

def calculate_travel_time(city1, city2, population_limit=200000):
    """Calculate the travel time based on distance rank, country, and population conditions."""
    base_times = [2, 4, 8]  # Time to first, second, and third nearest cities
    time = base_times[city2['rank']]
    
    # Extra time if crossing borders
    if city1['country'] != city2['country']:
        time += 2
    
    # Extra time if the destination city has a population > 200,000
    if city2['population'] > population_limit:
        time += 2

    return time

#%%
# --- Visualization Functions ---

def plot_route(route):
    """Plot the travel route on a 2D map."""
    lats, lons = zip(*[(city['latitude'], city['longitude']) for city in route])
    plt.plot(lons, lats, marker='o', color='b', linestyle='-')
    plt.title("World Travel Route")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
#%%
# --- Data Export Function ---

def save_summary(data, filename):
    """Save travel summary data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file)

#%%
# --- Main Program ---

def main():
    # Load city data
    data = load_data('data/worldcitiespop.csv')

    # Initialize journey from London
    starting_city = data[data['city'] == 'London'].iloc[0]
    journey = [starting_city]
    total_time = 0

    # Simulate journey around the world
    current_city = starting_city
    while total_time < 1920:  # 1920 hours = 80 days
        nearest_cities = find_nearest_cities(data, current_city)
        next_city = nearest_cities[0]  # Move to the nearest city
        travel_time = calculate_travel_time(current_city, next_city)
        total_time += travel_time
        journey.append(next_city)

        # Update the current city
        current_city = next_city

        # Check if back in London
        if current_city['city'] == 'London' and total_time <= 1920:
            break

    print("Total travel time:", total_time, "hours")
    plot_route(journey)
    save_summary({"total_time": total_time, "route": journey}, "output/travel_summary.json")

if __name__ == "__main__":
    main()
