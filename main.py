import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import folium

# %%
# --- Data Loading and Processing Functions ---


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

    # Sort by distance and get the top nearest cities
    nearest_cities = sorted(distances, key=lambda x: x[1])[:num_nearest]
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


# %%
# --- Travel Logic Functions ---


def calculate_travel_time(current_city, nearest_cities, population_limit=200000):
    """
    Calculate travel times to the 3 nearest cities based on the last selected city.
    - An additional 2 hours if the destination city is in another country than the starting city.
    - An additional 2 hours if the destination city has more than 200,000 inhabitants.
    """
    print(f"\nCalculation of travel time from {current_city['City']}:")

    base_times = [2, 4, 8]  # Time to first, second, and third nearest cities

    # Calculate the time to each of the nearest cities
    for city2_data in nearest_cities:
        time = base_times[
            city2_data["Rank"]
        ]  # Time based on the rank of the city (0, 1, 2)

        # Extra time if crossing borders
        if current_city["Country"] != city2_data["Country"]:
            time += 2
            print(
                f"Note: Crossing border adds 2 hours for travel to {city2_data['City']}."
            )

        # Extra time if the destination city has a population > 200,000
        if city2_data["Population"] > population_limit:
            time += 2
            print(
                f"Note: Large population (>200,000) adds 2 hours for travel to {city2_data['City']}."
            )

        print(
            f"- To {city2_data['City']} ({city2_data['Country']}): {time} hours "
            f"(Population: {city2_data['Population']})."
        )


def can_travel_around_the_world(data, start_city_name, population_limit=200000, latitude_tolerance=0.01):
    """
    Determine if it's possible to travel around the world starting from a user-specified city,
    always traveling east, and returning to the starting city. Calculates the full journey
    before checking if it fits within the 80-day limit.
    """
    # Ensure Population is numeric
    data["Population"] = pd.to_numeric(data["Population"], errors="coerce")

    DAYS_IN_HOURS = 80 * 24  # Convert 80 days to hours

    # Request starting city from the user
    if start_city_name not in data["City"].values:
        print(f"City '{start_city_name}' not found in the data.")
        return False, None, None, None, None, None

    total_time = 0
    current_city = data[data["City"] == start_city_name].iloc[0]
    start_city = current_city
    visited_cities = set()
    visited_countries = set()
    path = []

    # Storing statistics for charts
    country_times = {} # Time spent in different countries
    stage_times = {
        "Travel": 0,
        "Border Crossing": 0,
        "Stops": 0,
    }  # Time on the stages of the journey
    route_times = []

    print(
        f"\nStarting the journey from {current_city['City']} in {current_city['Country']}."
    )
    
    # Add time for the start city (we assume some base time, like 0 hours or the first time interval)
    country_times[current_city["Country"]] = stage_times["Travel"]

    while True:
        # Mark current city as visited and add it to the path
        visited_cities.add(current_city["City"])
        visited_countries.add(current_city["Country"])
        path.append(current_city["City"])
        
        # Add time spent in the current country to country_times
        #if current_city["Country"] not in country_times:
            #country_times[current_city["Country"]] = 0
        # Add time spent in the country (this is based on the time for this stage)   
        #country_times[current_city["Country"]] += stage_times["Travel"]

        # Check if we've returned to the starting city (after visiting at least one other city)
        if len(path) > 1 and current_city["City"] == start_city_name:
            print(
                f"Returned to {start_city_name} after visiting {len(visited_cities)} cities in {len(visited_countries)} countries.."
            )
            break

        # Find the nearest city to the east
        eastern_cities = data.loc[
            (data["Longitude"] > current_city["Longitude"])
            & (abs(data["Latitude"] - current_city["Latitude"]) <= latitude_tolerance)
            & (~data["City"].isin(visited_cities))
        ].copy()

        if eastern_cities.empty:
          # If no more eastern cities, wrap around to the western hemisphere
            eastern_cities = data.loc[
                (abs(data["Latitude"] - current_city["Latitude"]) <= latitude_tolerance)
                & (~data["City"].isin(visited_cities))
            ].copy()

        # If all cities are visited, break
        if eastern_cities.empty:
            print("No unvisited cities remaining.")
            eastern_cities = pd.DataFrame([start_city])

        # Find the nearest city by distance
        eastern_cities["Distance"] = eastern_cities.apply(
            lambda row: haversine(
                current_city["Latitude"],
                current_city["Longitude"],
                row["Latitude"],
                row["Longitude"],
            ),
            axis=1,
        )
        next_city = eastern_cities.loc[eastern_cities["Distance"].idxmin()]

        # Calculate travel time to the next city
        base_times = [2, 4, 8]
        rank = 0  # Always consider the nearest city as the first rank
        time = base_times[rank]
        
        current_country = current_city["Country"]
        
        if current_country in country_times:
            country_times[current_country] += time
        else:
            country_times[current_country] = time
        
        if current_city["Country"] != next_city["Country"]:
            print(
                f"Border crossing: {current_city['City']} -> {next_city['City']} ({next_city['Country']})"
            )
            time += 2  # Crossing borders adds 2 hours
            stage_times["Border Crossing"] += 2

        if (
            pd.notna(next_city["Population"])
            and float(next_city["Population"]) > 200000
        ):  # float(population_limit)
            time += 2  # Large population adds 2 hours
            stage_times["Stops"] += 2

        total_time += time
        stage_times["Travel"] += time
        current_city = next_city  # Update current city
        
        # Check if we've returned to the starting city
        if current_city["City"] == start_city_name:
            print(
                f"Returned to {start_city_name} after visiting {len(visited_cities)} cities."
            )
            break

    print(
        f"\nJourney completed in {total_time:.2f} hours ({total_time / 24:.2f} days)."
    )
    print(
        f"Visited {len(visited_cities)} cities in {len(visited_countries)} countries."
    )

    # Check if the journey was completed within the time limit
    if total_time <= DAYS_IN_HOURS:
        print("Journey completed within 80 days!")
        return True, route_times, total_time, path, country_times, stage_times
    else:
        print("Journey exceeded 80 days. Not possible to travel around the world within the time limit.")
        return False, route_times, total_time, path, country_times, stage_times


# %%
# --- Visualization Functions ---
def plot_route_on_map(route, current_city):
    """Plot the travel route on an interactive map using folium."""
    if len(route) < 1:
        print("There are not enough cities to build a route.")
        return

    # Center the map on the current city
    map_center = [current_city["Latitude"], current_city["Longitude"]]
    travel_map = folium.Map(location=map_center, zoom_start=6)

    # Add the current city to the map
    folium.Marker(
        location=[current_city["Latitude"], current_city["Longitude"]],
        popup=current_city["City"],
        tooltip=current_city["City"],
    ).add_to(travel_map)

    coordinates = []

    # For each nearest city, draw a line from the current city
    for city in route:
        coords = [city["Latitude"], city["Longitude"]]
        coordinates.append(coords)
        folium.Marker(location=coords, popup=city["City"], tooltip=city["City"]).add_to(
            travel_map
        )

        # Draw the route line between current city and nearest city
        folium.PolyLine(
            locations=[
                [current_city["Latitude"], current_city["Longitude"]],
                [city["Latitude"], city["Longitude"]],
            ],
            color="blue",
            weight=2.5,
            opacity=0.7,
        ).add_to(travel_map)

    # Save map as HTML file
    travel_map.save("travel_route_map.html")
    print(
        "\nThe route is saved in a file travel_route_map.html. Open this file in a browser."
    )



# %%
# --- Visualization Functions 2 ---
def plot_pie_chart(country_times):
    """
    Построить круговую диаграмму времени, проведённого в каждой стране.
    """
    labels = list(country_times.keys())
    sizes = list(country_times.values())

    # Define colors for the pie chart (you can customize this list as needed)
    colors = ["#ff9999", "#66b3ff", "#ffcc99", "#ffb3e6", "#c2c2f0"]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
    plt.title("Time Spent in Each Country")
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plot_stacked_bar_chart(stage_times):
    """
    Plot a bar chart comparing cities based on total travel time and cities visited.
    """
    travel_times = stage_times["Travel"] 
    border_delays = stage_times["Border Crossing"] 
    other_delays = stage_times["Stops"]

    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    
    ax.bar(0, travel_times, bar_width, label="Total Travel Time", color="skyblue")
    ax.bar(1, border_delays, bar_width, label="Border Delay", color="orange")
    ax.bar(2, other_delays, bar_width, label="Large City Delay", color="gray")

    ax.set_xlabel("Stages")
    ax.set_ylabel("Time (hours)")
    ax.set_title("Travel Time by Stage")
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Total Travel Time","Border Delay","Large City Delay"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()

# %%
# --- Data Export Function ---


def save_summary(data, filename):
    """Save travel summary data to a JSON file."""
    with open(filename, "w") as file:
        json.dump(data, file)


# %%
# --- Main Program ---

def main():
    # Load city data
    data = load_data("data/worldcitiespop.csv")
    print("The first rows of data:")
    print(data.head())  # first lines output

    country_times, stage_times = None, None

    while True:  # Requesting the city from the user
        city_name = input(
            "\nEnter the name of the city to search for the nearest: "
        ).strip()

        if city_name in data["City"].values:
            current_city = data[data["City"] == city_name].iloc[0]

            nearest_cities = find_nearest_cities(
                data, current_city
            )  # Find 3 nearest cities

            # Plot route on the map
            plot_route_on_map(nearest_cities, current_city)

            # РWe calculate the time to all 3 nearest cities
            calculate_travel_time(current_city, nearest_cities)
        else:
            print(f"City '{city_name}' not found in the data.")
            continue

        # Checking if the user wants to continue
        user_response = (
            input("\nDo you want to check out another city? (Enter 'yes' or 'no'):")
            .strip()
            .lower()
        )

        if user_response == "no":
            print("Completion of the program.")
            break
        elif user_response == "yes":
            continue  # Loop restarts, asking for a new city
        else:
            # Keep asking until the user provides a valid answer
            while user_response not in ["yes", "no"]:
                print("Please enter 'yes' or 'no'.")
                user_response = (
                    input(
                        "Do you want to check out another city? (Enter 'yes' or 'no'): "
                    )
                    .strip()
                    .lower()
                )

            if user_response == "no":
                print("Completion of the program.")
                break
            elif user_response == "yes":
                continue

    while True:  # Check if the user wants to calculate a round-the-world journey

        round_the_world_choice = (
            input(
                "\nDo you want to calculate a round-the-world journey? (Enter 'yes' or 'no'): "
            )
            .strip()
            .lower()
        )

        if round_the_world_choice == "no":
            print("Completion of the program.")
            break  # Exit the loop if the user chooses "no"
        elif round_the_world_choice == "yes":
            while True:
                city_name = input(
                    "Enter the name of the starting city for the round-the-world journey: "
                ).strip()

                if city_name in data["City"].values:
                    journey_possible, route_times, total_time, path, country_times, stage_times = (
                        can_travel_around_the_world(data, city_name)
                    )

                    if journey_possible:
                        print(f"Path taken: {' -> '.join(path)}")
                    else:
                        print("\nIt's not possible to travel around the world within 80 days.")
                    break
                else:
                    print(f"City '{city_name}' not found in the data.")
                # Если город найден, выходим из внутреннего цикла
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            # Don't continue asking until the user gives valid input

    user_input = (
        input("\nWould you like to plot visualizations? (Enter 'yes' or 'no'): ")
        .strip()
        .lower()
    )

    if user_input == "yes":

        if country_times:
            plot_pie_chart(country_times)
        else:
            print("No country time data available for visualization.")

        if stage_times:
            plot_stacked_bar_chart(stage_times)
        else:
            print("No stage time data available for visualization.")

    print("Program completed.")

if __name__ == "__main__":
    main()