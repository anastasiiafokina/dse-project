# --- Main Program ---
from modules.data_processing import load_data, find_nearest_cities
from modules.travel_logic import calculate_travel_time, can_travel_around_the_world
from modules.visualization import plot_route_on_map, plot_pie_chart, plot_stacked_bar_chart

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
                    input("Do you want to check out another city? (Enter 'yes' or 'no'): ")
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
            input("\nDo you want to calculate a round-the-world journey? (Enter 'yes' or 'no'): ")
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
                    journey_possible, path, country_times, stage_times = (
                        can_travel_around_the_world(data, city_name)
                    )

                    if journey_possible:
                        print(f"Path taken: {' -> '.join(path)}")
                    else:
                        print("\nIt's not possible to travel around the world within 80 days.")
                    break
                else:
                    print(f"City '{city_name}' not found in the data.")
                # If the city is found, we exit the inner loop
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