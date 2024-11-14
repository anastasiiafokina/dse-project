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
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) #angular distance between two points, expressed in radians
    return R * c  # Distance in kilometers

def calculate_distance_between_cities(city1_name, city2_name, data):
    """Calculate the distance between two cities by their names."""
    city1 = data[data['City'] == city1_name].iloc[0]
    city2 = data[data['City'] == city2_name].iloc[0]
    
    # Получение координат городов
    lat1, lon1 = city1['Latitude'], city1['Longitude']
    lat2, lon2 = city2['Latitude'], city2['Longitude']
    
    # Рассчитываем расстояние с помощью функции haversine
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
    
    # Сортировка списка расстояний и выбор ближайших городов
    distances.sort(key=lambda x: x[1])
    return distances[:num_nearest]
    # Вывод ближайших городов и расстояний
    print(f"Ближайшие {num_nearest} города к {current_city['City']}:")
    for city, dist in nearest_cities:
        print(f"{city['City']} на расстоянии {dist:.2f} км")
        

    # Sort by distance and get the top nearest cities
    nearest_cities = sorted(distances, key=lambda x: x[1])[:num_nearest]
    # Add rank to each nearest city (0 for nearest, 1 for second nearest, etc.)
    return [{'City': city['City'], 'Country': city['Country'], 'Population': city['Population'],
             'Latitude': city['Latitude'], 'Longitude': city['Longitude'], 'Rank': rank}
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
    print(data.head()) #first lines output 
    
    # Запрос названия города от пользователя
    city_name = input("Введите название города для поиска ближайших: ").strip()
    
    # Проверка на наличие введенного города в данных
    if city_name in data['City'].values:
        # Находим текущий город в данных
        current_city = data[data['City'] == city_name].iloc[0]
        # Выводим ближайшие города
        find_nearest_cities(data, current_city)
    else:
        print(f"Город '{city_name}' не найден в данных.")
        

    # Initialize journey from London
    starting_city = data[data['City'] == 'london'].iloc[0]
    journey = [starting_city]
    total_time = 0
    
    # Запрос названий городов от пользователя
    city1 = input("Enter the name of the first city: ").strip()
    city2 = input("Enter the name of the second city: ").strip()
    
    # Проверка на наличие введенных городов в данных
    if city1 in data['City'].values and city2 in data['City'].values:
        calculate_distance_between_cities(city1, city2, data)
    else:
        print("One or both of the specified cities were not found in the data.")


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
