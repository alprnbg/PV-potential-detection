import numpy as np
import math
from pyproj import Transformer

# Information extracted from the dataset header
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000
NODATA_VALUE = -999

# Load data as 2d array
data = np.loadtxt("grids_germany_annual_radiation_global_2022.asc", skiprows=28)
data[data == -999] = np.nan

# Define coordinate systems
from_crs = "EPSG:4326"  # WGS 84
to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

# Create transformer object
transformer = Transformer.from_crs(from_crs, to_crs)

def calculate_kwh(area, latitude, longitude, is_south):
    try:
        # Convert latitude and longitude to Gauss Krüger coordinates
        h, r = transformer.transform(latitude, longitude)

        y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil((h - YLLCORNER) / CELLSIZE)
        radiance = data[x, y]
    except:
        radiance = 0


    potential_output = area * radiance * 0.15 * 0.75

    if not is_south:
        potential_output *= 0.85
    return potential_output

# Calculate potential output for all roofs in roof_houses_all.csv
import pandas as pd
import ast

df_path = "roof_oberbayern-houses-orientation.csv"
roof_df = pd.read_csv(df_path)
potential_output = []
for i, row in roof_df.iterrows():
    area = row["area"]
    long = ast.literal_eval(row["center"])[0]
    lat = ast.literal_eval(row["center"])[1]
    is_south = row["orientation"]
    out = calculate_kwh(area, lat, long, is_south)
    potential_output.append(out)
roof_df["potential_output"] = potential_output
roof_df.to_csv("roof_houses_all_output.csv", index=False)

# sort by potential output
roof_df.sort_values(by=["potential_output"], ascending=False, inplace=True)
# save top 100
roof_df.head(100).to_csv("top_100.csv", index=False)