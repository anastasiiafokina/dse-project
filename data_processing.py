# modules/data_processing.py
# Load city data, preprocess it, and calculate distances.
import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def haversine(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate great-circle distance between points
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # Distance in kilometers
