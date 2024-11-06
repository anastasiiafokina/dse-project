# main.py
# Run the journey by importing modules and calculating total travel time.

from modules.data_processing.py import load_data, haversine
from modules.travel_logic import calculate_travel_time
from modules.visualization import plot_route
from modules.data_import_export import save_summary

def main():
    data = load_data('data/cities.csv')
    # Assume some logic here to calculate nearest cities and create the journey
    journey = []
    total_time = 0

    # Example route loop
    for city in journey:
        # Assume 'next_city' is calculated based on rules
        travel_time = calculate_travel_time(city, next_city)
        total_time += travel_time
        journey.append(next_city)

    print("Total travel time:", total_time)
    plot_route(journey)
    save_summary({"total_time": total_time, "route": journey}, "output/travel_summary.json")

if __name__ == "__main__":
    main()
