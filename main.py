import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import folium

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
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) #angular distance between two points, expressed in radians
    return R * c  # Distance in kilometers

def calculate_distance_between_cities(city1_name, city2_name, data):
    """Calculate the distance between two cities by their names."""
    city1 = data[data['City'] == city1_name].iloc[0]
    city2 = data[data['City'] == city2_name].iloc[0]
    
    # Getting coordinates of cities
    lat1, lon1 = city1['Latitude'], city1['Longitude']
    lat2, lon2 = city2['Latitude'], city2['Longitude']
    
    # Calculate the distance using the haversine function
    distance = haversine(lat1, lon1, lat2, lon2)
    
    print(f"Distance between {city1_name} and {city2_name}: {distance:.2f} km")
    return distance

def find_nearest_cities(data, current_city, num_nearest=3):
    """Find the 3 nearest cities to the current city, based on latitude and longitude."""
    distances = []
    for _, city in data.iterrows():
        if city['City'] != current_city['City']:  # Skip the same city
            dist = haversine(current_city['Latitude'], current_city['Longitude'], city['Latitude'], city['Longitude'])
            distances.append((city, dist))
    
    # Sorting the list of distances and selecting the nearest cities
    distances.sort(key=lambda x: x[1])
    nearest_cities = distances[:num_nearest]
    
    # Display nearest cities and distances
    print(f"\nThe nearest {num_nearest} cities to {current_city['City']}:")
    for city, dist in nearest_cities:
        print(f"-{city['City']} is {dist:.2f} km away")
        

    # Sort by distance and get the top nearest cities
    nearest_cities = sorted(distances, key=lambda x: x[1])[:num_nearest]
    # Add rank to each nearest city (0 for nearest, 1 for second nearest, etc.)
    return [{'City': city['City'], 'Country': city['Country'], 'Population': city['Population'],
             'Latitude': city['Latitude'], 'Longitude': city['Longitude'], 'Rank': rank}
            for rank, (city, _) in enumerate(nearest_cities)]
    
#%%
# --- Travel Logic Functions ---

def calculate_travel_time(current_city, nearest_cities, population_limit=200000):
    """
    Calculate travel times to the 3 nearest cities based on the last selected city.
    """
    print(f"\nCalculation of travel time from {current_city['City']}:")

    base_times = [2, 4, 8]  # Time to first, second, and third nearest cities

    # Calculate the time to each of the nearest cities
    for city2_data in nearest_cities:
        time = base_times[city2_data['Rank']]  # Time based on the rank of the city (0, 1, 2)

        # Extra time if crossing borders
        if current_city['Country'] != city2_data['Country']:
            time += 2

        # Extra time if the destination city has a population > 200,000
        if city2_data['Population'] > population_limit:
            time += 2

        print(
            f"- To {city2_data['City']} ({city2_data['Country']}): {time} hours "
            f"(Population: {city2_data['Population']})."
        )

#%%
# --- Visualization Functions ---
def plot_route_on_map(route, current_city):
    """Plot the travel route on an interactive map using folium."""
    if len(route) < 1:
        print("There are not enough cities to build a route.")
        return

    # Center the map on the current city
    map_center = [current_city['Latitude'], current_city['Longitude']]
    travel_map = folium.Map(location=map_center, zoom_start=6)

    # Add the current city to the map
    folium.Marker(
        location=[current_city['Latitude'], current_city['Longitude']],
        popup=current_city['City'],
        tooltip=current_city['City']
    ).add_to(travel_map)

    coordinates = []

    # For each nearest city, draw a line from the current city
    for city in route:
        coords = [city['Latitude'], city['Longitude']]
        coordinates.append(coords)
        folium.Marker(location=coords, popup=city['City'], tooltip=city['City']).add_to(travel_map)

        # Draw the route line between current city and nearest city
        folium.PolyLine(
            locations=[[current_city['Latitude'], current_city['Longitude']], [city['Latitude'], city['Longitude']]],
            color="blue", weight=2.5, opacity=0.7
        ).add_to(travel_map)

    # Save map as HTML file
    travel_map.save('travel_route_map.html')
    print("\nThe route is saved in a file travel_route_map.html. Open this file in a browser.")

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
    print("The first rows of data:")
    print(data.head()) #first lines output 
    
    
    while True:
        # Requesting the city from the user
        city_name = input("\nEnter the name of the city to search for the nearest: ").strip()
        
        if city_name in data['City'].values:
            current_city = data[data['City'] == city_name].iloc[0]
            
            nearest_cities = find_nearest_cities(data, current_city)  # Find 3 nearest cities

            # Plot route on the map
            plot_route_on_map(nearest_cities, current_city)

            # РWe calculate the time to all 3 nearest cities
            calculate_travel_time(current_city, nearest_cities)
        else:
            print(f"City '{city_name}' not found in the data.")
            continue
 
        # Checking if the user wants to continue
        user_response = input("\nDo you want to check out another city? (Enter 'yes' or 'no'):").strip().lower()
        
        if user_response == "no":
            print("Completion of the program.")
            break
        elif user_response == "yes":
            continue  # Loop restarts, asking for a new city
        else:
            # Keep asking until the user provides a valid answer
            while user_response not in ["yes", "no"]:
                print("Please enter 'yes' or 'no'.")
                user_response = input("Do you want to check out another city? (Enter 'yes' or 'no'): ").strip().lower()

            if user_response == "no":
                print("Completion of the program.")
                break
            elif user_response == "yes":
                continue
    


if __name__ == "__main__":
    main()
