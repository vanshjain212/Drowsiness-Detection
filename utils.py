import numpy as np
import csv
from datetime import datetime

LOG_FILE = "logs.csv"

# Initialize CSV with Header (Run once when imported)
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Event", "Value", "Threshold"])

def euclidean(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def log_event(event_type, value, threshold):
    """Saves event data to a CSV for post-drive analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, f"{value:.3f}", f"{threshold:.3f}"])