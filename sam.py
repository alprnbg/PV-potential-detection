import os
import ast

import cv2
import numpy as np
import pandas as pd
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

from map_utils import getPointXY, get_satellite_img, get_roof_mask, intersection_and_union

ZOOM = 20
IMSIZE = 400


class SAM:
    def __init__(self):
        sam = sam_model_registry["vit_b"](checkpoint="model/sam_vit_b_01ec64.pth").cuda()
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
    def inference(self, rgb_img, poly_points):
        masks = self.mask_generator.generate(rgb_img)
        osm_mask = get_roof_mask(IMSIZE, poly_points)
        filtered, whole_roof = filter_masks(masks, osm_mask)
        return filtered, whole_roof


def draw_segments(img, anns, whole_roof):
    segment_overlay = None
    whole_roof_overlay = None
    if len(anns)>0:
        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
        for filtered_mask in anns:
            m = filtered_mask["segmentation"]
            color_mask = np.random.random((1, 3))[0]
            mask[m, :] = list(map(int, color_mask*255))
        segment_overlay = cv2.addWeighted(mask, 1, img, 1, 0, segment_overlay)
    if whole_roof:
        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
        color_mask = np.random.random((1, 3))[0]
        mask[whole_roof["segmentation"], :] = list(map(int, color_mask*255))
        whole_roof_overlay = cv2.addWeighted(mask, 0.4, img, 0.8, 0, whole_roof_overlay)
    return segment_overlay, whole_roof_overlay
        

def filter_masks(masks, osm_mask, iou_thresh=0.20):
    filtered = []
    whole_roof = None
    for mask in masks:
        inters, union = intersection_and_union(osm_mask, mask["segmentation"])
        iou = inters/union
        if iou > iou_thresh:
            mask_area = mask["segmentation"].sum()
            osm_area = osm_mask.sum()
            if abs(mask_area - osm_area)/osm_area < 0.2:
                whole_roof = mask
            else:
                filtered.append(mask)
    return filtered, whole_roof


def process_image(sam, center_coords, poly, b_id):
    print(b_id)
    image = get_satellite_img(*center_coords, ZOOM, IMSIZE)
    #image = cv2.imread("/home/alperen/Workspace/photongraphy/images/raw/oberbayern-houses_2.jpg")
    points = getPointXY(center_coords[1], center_coords[0], ZOOM, np.array(poly)[:,0], np.array(poly)[:,1], IMSIZE)
    proj_roof_image = image.copy()
    cv2.drawContours(proj_roof_image, [np.array(points).astype(int).T], -1, (0, 255, 0), 2)
    
    rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    segments, whole_roof = sam.inference(rgb_image, points)
    seg_img, whole_roof_img = draw_segments(image.copy(), segments, whole_roof)
    print("len segments:", len(segments))
    print(seg_img is None, whole_roof_img is None)
    is_south = "Not detected"
    if len(segments) > 0:
        orientations, centers, is_south = get_orientations(segments)
        if len(segments) > 1:
            if is_south:
                is_south = "SN"
            else:
                is_south = "EW"
        orient_img = seg_img.copy()
        for orient, center in zip(orientations, centers):
            orient_img = cv2.arrowedLine(orient_img, list(map(int, center)), [int(center[0]+orient[0]),
                                                              int(center[1]+orient[1])] 
                                         ,(0,0,255), 2) 
        cv2.imwrite(os.path.join("images", "orientation", f"{b_id}.jpg"), orient_img)
    cv2.imwrite(os.path.join("images", "raw", f"{b_id}.jpg"), image)
    cv2.imwrite(os.path.join("images", "roof_images", f"{b_id}.jpg"), proj_roof_image)
    if seg_img is not None:
        cv2.imwrite(os.path.join("images", "segments", f"{b_id}.jpg"), seg_img)
    if whole_roof_img is not None:
        if len(segments) == 0:
            # todo flat roof
            pass
        cv2.imwrite(os.path.join("images", "whole_roof", f"{b_id}.jpg"), whole_roof_img)
    return is_south


def get_orientations(segms):
    centers = []
    for seg in segms:
        mask = seg["segmentation"]
        contours, _ = cv2.findContours(mask.astype("uint8")*255, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        M = cv2.moments(cnt)
        x = round(M['m10'] / M['m00'])
        y = round(M['m01'] / M['m00'])
        centers.append([x,y])
    centers = np.array(centers)
    x_mean, y_mean = centers[:,0].mean(), centers[:,1].mean()
    directions = [(x-x_mean, y-y_mean) for x,y in centers]
    is_south = [abs(y)>abs(x) for x,y in directions]
    return directions, centers, any(is_south)


if __name__ == "__main__":
    
    df_path = "csv/roof_houses_all.csv"
    
    os.makedirs(os.path.join("images", "raw"), exist_ok=True)
    os.makedirs(os.path.join("images", "roof_images"), exist_ok=True)
    os.makedirs(os.path.join("images", "segments"), exist_ok=True)
    os.makedirs(os.path.join("images", "whole_roof"), exist_ok=True)
    os.makedirs(os.path.join("images", "orientation"), exist_ok=True)
    sam = SAM()
    roof_df = pd.read_csv(df_path)
    orientations = [None]*len(roof_df)
    
    indices = list(range(50000))
    np.random.shuffle(indices)
    
    for index in tqdm(indices, total=len(indices)):
        try:
            row = roof_df.iloc[index]
            center_coords = ast.literal_eval(row["center"])
            poly = ast.literal_eval(row["poly"])
            b_id = row["id"]
            is_south = process_image(sam, center_coords, poly, b_id)
            orientations[index] = is_south
        except KeyboardInterrupt:
            break
    roof_df["orientation"] = orientations
    roof_df.to_csv(df_path.split(".")[0]+"_orientations.csv", index=False)

