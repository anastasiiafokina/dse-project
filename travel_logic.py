# modules/travel_logic.py
# Calculate travel time between cities based on proximity and conditions.
def calculate_travel_time(city1, city2, population_limit=200000):
    base_times = [2, 4, 8]  # Time to first, second, and third nearest cities
    time = base_times[city2['rank']]
    
    # Extra time if crossing borders
    if city1['country'] != city2['country']:
        time += 2
    
    # Extra time if city population > 200,000
    if city2['population'] > population_limit:
        time += 2

    return time
