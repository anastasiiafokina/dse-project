# modules/data_import_export.py
# Manage data import and export (e.g., outputting travel summaries).
import json

def save_summary(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)
