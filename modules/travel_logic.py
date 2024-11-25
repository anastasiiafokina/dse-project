import pandas as pd
from modules.data_processing import haversine


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
            print(f"Border crossing: {current_city['City']} -> {next_city['City']} ({next_city['Country']})")
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

    print(f"\nJourney completed in {total_time:.2f} hours ({total_time / 24:.2f} days).")
    print(f"Visited {len(visited_cities)} cities in {len(visited_countries)} countries.")

    # Check if the journey was completed within the time limit
    if total_time <= DAYS_IN_HOURS:
        print("Journey completed within 80 days!")
        return True, path, country_times, stage_times
    else:
        print("Journey exceeded 80 days. Not possible to travel around the world within the time limit.")
        return False, path, country_times, stage_times
