from pyproj import Geod
import numpy as np
import ast

def get_roof_area(poly_shape):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(poly_shape)[0])
    return area


def get_roof_orientation(tags):
    if tags:
        tags = ast.literal_eval(tags)
        if "roof:direction" in tags:
            return tags["roof:direction"]
        elif "roof:orientation" in tags:
            return tags["roof:orientation"]
    return np.nan
