import matplotlib.pyplot as plt
import folium

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
    Create a pie chart of the time spent in each country.
    """
    labels = list(country_times.keys())
    sizes = list(country_times.values())

    # Define colors for the pie chart (you can customize this list as needed)
    colors = ["#4A6D91", "#F1E1C6", "#B5B8B1", "#A3C4F3", "#D8C7A1"]
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

    # Creating a graph
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    
    ax.bar(0, travel_times, bar_width, label="Total Travel Time", color="#4A6D91")
    ax.bar(1, border_delays, bar_width, label="Border Delay", color="#D8C7A1")
    ax.bar(2, other_delays, bar_width, label="Large City Delay", color="#B5B8B1")

    ax.set_xlabel("Stages")
    ax.set_ylabel("Time (hours)")
    ax.set_title("Travel Time by Stage")
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Total Travel Time","Border Delay","Large City Delay"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
