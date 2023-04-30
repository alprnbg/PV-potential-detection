import numpy as np
import requests
import math
import cv2

API_KEY=""


def get_satellite_img(lon, lat, zoom, imsize):
    url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&maptype=satellite&size={}x{}&key={}".format(lat, lon, zoom, imsize, imsize, API_KEY)
    response = requests.get(url).content
    image = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_UNCHANGED)
    return image


def getPointXY(c_lat, c_lng, zoom, pointLng, pointLat, imsize):
    parallelMultiplier = math.cos(c_lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    y = (c_lat - pointLat)/degreesPerPixelY + imsize / 2 
    x = (pointLng - c_lng)/degreesPerPixelX + imsize / 2
    return (x, y)


def get_roof_mask(imsize, poly_points):
    mask = np.zeros((imsize, imsize))
    cv2.drawContours(mask, [np.array(poly_points).astype(int).T], -1, (255), -1)
    return mask.astype("bool")


def intersection_and_union(mask1, mask2):
    return (mask1 & mask2 ).sum(), (mask1 | mask2).sum()
