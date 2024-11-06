# modules/visualization.py
# Create a map plot showing the route between cities.
import matplotlib.pyplot as plt

def plot_route(route):
    lats, lons = zip(*[(city['latitude'], city['longitude']) for city in route])
    plt.plot(lons, lats, marker='o', color='b', linestyle='-')
    plt.title("World Travel Route")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
