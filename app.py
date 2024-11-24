import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# --- Functions ---
# Вставьте сюда все ваши функции, которые вы описали в основном коде.
# Например:
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
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # Distance in kilometers

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
    distances.sort(key=lambda x: x[1])
    nearest_cities = distances[:num_nearest]
    return [
        {
            "City": city["City"],
            "Country": city["Country"],
            "Population": city["Population"],
            "Latitude": city["Latitude"],
            "Longitude": city["Longitude"],
            "Distance": dist,
        }
        for city, dist in distances[:num_nearest]
    ]
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
        return (True, route_times, total_time, path, country_times, stage_times)
    else:
        print("Journey exceeded 80 days. Not possible to travel around the world within the time limit.")
        return (False, route_times, total_time, path, country_times, stage_times)

def plot_route_on_map(route, current_city):
    """Plot the travel route on an interactive map using folium."""
    map_center = [current_city["Latitude"], current_city["Longitude"]]
    travel_map = folium.Map(location=map_center, zoom_start=6)
    folium.Marker(
        location=[current_city["Latitude"], current_city["Longitude"]],
        popup=current_city["City"],
        tooltip=current_city["City"],
        icon=folium.Icon(color="red"),
    ).add_to(travel_map)
    for city in route:
        folium.Marker(
            location=[city["Latitude"], city["Longitude"]],
            popup=city["City"],
            tooltip=f"{city['City']} ({city['Distance']:.2f} km)",
        ).add_to(travel_map)
        folium.PolyLine(
            locations=[
                [current_city["Latitude"], current_city["Longitude"]],
                [city["Latitude"], city["Longitude"]],
            ],
            color="blue",
            weight=2.5,
            opacity=0.7,
        ).add_to(travel_map)
    return travel_map

def plot_pie_chart(country_times):
    """
    Create a pie chart of the time spent in each country.
    """
    
   
    if not country_times:
        st.write("Pie chart data is empty.")
        return None

    labels = list(country_times.keys())
    sizes = list(country_times.values())
    colors = ["#4A6D91", "#F1E1C6", "#B5B8B1", "#A3C4F3", "#D8C7A1"]
    #colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
    ax.set_title("Time Spent in Each Country")
    return fig



def plot_stacked_bar_chart(stage_times):
    """
    Plot a bar chart comparing cities based on total travel time and cities visited.
    """
    
    
    if not stage_times:
        st.write("Stacked bar chart data is empty.")
        return None

    travel_times = stage_times.get("Travel", 0)
    border_delays = stage_times.get("Border Crossing", 0)
    other_delays = stage_times.get("Stops", 0)

    if travel_times == 0 and border_delays == 0 and other_delays == 0:
        st.write("All values in stage_times are zero.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8

    ax.bar(0, travel_times, bar_width, label="Total Travel Time", color="#4A6D91")
    ax.bar(1, border_delays, bar_width, label="Border Delay", color="#D8C7A1")
    ax.bar(2, other_delays, bar_width, label="Large City Delay", color="#B5B8B1")

    ax.set_xlabel("Stages")
    ax.set_ylabel("Time (hours)")
    ax.set_title("Travel Time by Stage")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Total Travel Time", "Border Delay", "Large City Delay"], rotation=45, ha="right")
    ax.legend()

    return fig




# --- Streamlit App ---
st.title("Travel Project")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your city CSV file", type=["csv"])

if uploaded_file:
    # Load data
    data = load_data(uploaded_file)
    if "City" not in data.columns:
        st.error("The uploaded file must contain a 'City' column.")
    else:
        st.success("Data successfully loaded!")
        city_names = data["City"].unique()

        # Select a city
        selected_city_name = st.sidebar.selectbox("Select a City", city_names)
        if selected_city_name:
            current_city = data[data["City"] == selected_city_name].iloc[0]
            st.write(f"### Current City: {selected_city_name}")
            st.map(pd.DataFrame([{
                "lat": current_city["Latitude"],
                "lon": current_city["Longitude"],
            }]))

            # Find nearest cities
            nearest_cities = find_nearest_cities(data, current_city)
            st.write("### Nearest Cities:")
            st.write("Calculate the distance and time to the 3 nearest cities.")
            st.write(pd.DataFrame(nearest_cities))
            
            if nearest_cities:
                # Добавление рангов для ближайших городов
                for i, city in enumerate(nearest_cities):
                    city["Rank"] = i  # 0 для самого ближайшего, 1 для второго и т.д.

                # Расчет времени до ближайших городов
                calculate_travel_time(current_city, nearest_cities)

                # Отображение времени на странице
                st.write("#### Travel Times to Nearest Cities:")
                travel_times = []
                base_times = [2, 4, 8]
                for city in nearest_cities:
                    travel_time = base_times[city["Rank"]]
                    extra_time = 0

                    # Дополнительное время за пересечение границ
                    if current_city["Country"] != city["Country"]:
                        travel_time += 2
                        extra_time += 2

                    # Дополнительное время за большое население
                    if city["Population"] > 200000:
                        travel_time += 2
                        extra_time += 2

                    travel_times.append({
                        "City": city["City"],
                        "Country": city["Country"],
                        "Base Time (hours)": base_times[city["Rank"]],
                        "Extra Time (hours)": extra_time,
                        "Total Time (hours)": travel_time,
                    })

                # Показ результатов в таблице
                st.write(pd.DataFrame(travel_times))

            

            # Plot route on map
            st.write('The map is displayed demonstrating the "path" from the main city to the neighbors.')
            travel_map = plot_route_on_map(nearest_cities, current_city)
            st_folium(travel_map, width=700, height=500)

            # Option to calculate round-the-world journey
            st.write("The program determines if a user can travel eastward continuously, visiting unvisited cities, and return to the starting point within 80 days (converted to hours).")
            if st.button("Calculate Round-the-World Journey"):
                
                result, route_times, total_time, path, country_times, stage_times = can_travel_around_the_world(data, selected_city_name)

                st.write("### Round-the-World Journey Results")
                if result:  # Теперь result[0] заменён на result
                    st.success("It's possible to complete a round-the-world journey!")
                else:
                    st.error("Round-the-world journey is not possible.")

                # Display additional results
                st.write(f"**Total Time:** {total_time:.2f} hours ({total_time / 24:.2f} days)")
                st.write("**Visited Cities in Order:**")
                st.write(path)
                
                # Display the pie chart of time spent in each country
                st.write("### Time Spent in Each Country")
                fig1 = plot_pie_chart(country_times)
                if fig1:
                    st.pyplot(fig1)
                else:
                    st.write("Pie chart not available.")  # Вывод сообщения, если график не построился

                # Display the stacked bar chart of time spent at different stages
                st.write("### Time Spent by Travel Stages")
                fig2 = plot_stacked_bar_chart(stage_times)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.write("Stacked bar chart not available.")  # Вывод сообщения, если график не построился


    

               
