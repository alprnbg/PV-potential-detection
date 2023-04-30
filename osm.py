import os

import pandas as pd
from tqdm import tqdm
from pyrosm import OSM
from shapely.geometry import shape
import numpy as np

from osm_utils import get_roof_area, get_roof_orientation


class OSMBuildingAnalyzer:
    def __init__(self, osm_file_path):
        self.name = osm_file_path.split(os.sep)[-1].split(".")[0]
        self.osm = OSM(osm_file_path)
        print("Reading building data from osm file")
        self.buildings = self.osm.get_buildings()
        for key, value in self.buildings.building.value_counts().items():
            print("Number of", key, ":", value)
        print("Total:", len(self.buildings))
        self.buildings["roof_orientation"] = self.buildings["tags"].apply(get_roof_orientation)
        print("Processing building data")
        self.roof_data = self._process_buildings()
        self.roof_data.to_csv("csv/roof_"+self.name+".csv", index=False)

    def _process_buildings(self):
        roof_data = {"center":[], "area":[], "orientation":[], "poly":[], "id":[]}
        indices = list(range(10000))
        np.random.shuffle(indices)
        for idx, index in tqdm(enumerate(indices), total=len(indices)):
            element = self.buildings.iloc[index]
            b_id = self.name + "_" + str(idx)
            # Get the geometry of the building
            building_shape = shape(element.geometry)
            # Get the roof area of the building
            roof_area = get_roof_area(building_shape)
            # Get the center coordinates of the building
            center_coords = building_shape.centroid.coords[0]
            roof_poly = building_shape.exterior.coords[:]
            # Add the data to the dictionary
            roof_data["center"].append(center_coords)
            roof_data["area"].append(roof_area)
            roof_data["orientation"].append(element["roof_orientation"])
            roof_data["poly"].append(roof_poly)
            roof_data["id"].append(b_id)
            if idx == 10000:
                break
        return pd.DataFrame(roof_data)


"""
baden-wuerttemberg.osm.pbf
berlin.osm.pbf
hamburg.osm.pbf
niedersachsen.osm.pbf
rheinland-pfalz.osm.pbf
sachsen-anhalt.osm.pbf
thueringen.osm.pbf
"""

if __name__=="__main__":
    building_analyzer = OSMBuildingAnalyzer(os.path.join('data','berlin-houses.osm.pbf'))
    