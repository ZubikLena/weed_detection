from __future__ import annotations
import os
from cv2 import CalibrateRobertson
import numpy as np
import cv2
import xml.etree.ElementTree as ET
BASE_DIR = "./data/synthetic/bounding_box"
IMG_DIR = "./data/synthetic/imgs"
OUTPUT_DIR = "./data/synthetic/images_cropped"

"""
ANNOTATION FORMAT EXAMPLE:
class x_center_normalized y_center_normalized width_normalized height_normalized
2.0 0.626563 0.214583 0.078125 0.104167
2.0 0.514844 0.037500 0.048438 0.050000
3.0 0.031250 0.091667 0.062500 0.183333
2.0 0.725781 0.000000 0.001563 0.000000 
2.0 0.034375 0.181250 0.068750 0.062500
2.0 0.418750 0.162500 0.075000 0.095833
3.0 0.011719 0.712500 0.023438 0.075000
3.0 0.138281 0.615625 0.251563 0.243750
2.0 0.398438 0.983333 0.059375 0.029167
3.0 0.399219 0.815625 0.157812 0.293750
2.0 0.873437 0.361458 0.043750 0.047917
3.0 0.741406 0.315625 0.182812 0.214583
3.0 0.560937 0.582292 0.143750 0.293750
"""

"""
class 3.0 is carrot
class 2.0 is weed
"""


def main():

    # Find all xml files in the directory
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.txt')]

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in files:
        img = cv2.imread(os.path.join(IMG_DIR, file.replace('.txt', '.png')))

        annotations = getBoundingBoxes(file, img)
        annotations = removeOutOfBounds(annotations, img)
        annotations = removeOverlap(annotations)
        if len(annotations) > 0:
            cropped_imgs = extract_bounding_boxes(annotations, img)
            for img in cropped_imgs:
                for key, value in img.items():
                    # Check if output directory exists
                    if not os.path.exists(os.path.join(OUTPUT_DIR, key)):
                        os.makedirs(os.path.join(OUTPUT_DIR, key))
                    # Save image with rising index
                    # check if img exists
                    if value.any():
                        cv2.imwrite(os.path.join(OUTPUT_DIR, key, str(
                            len(os.listdir(os.path.join(OUTPUT_DIR, key)))) + '.png'), value)


def getBoundingBoxes(file, img):
    # get img dimensions
    height, width, channels = img.shape
    # Get bounding boxes and labels
    bounding_boxes = []
    # bbox type: {'x': x, 'y': y, 'width': width, 'height': height, 'label': label}
    bboxes = []
    # go over each line in the txt file and extract all numbers that are delimited by a space
    for line in open(os.path.join(BASE_DIR, file)):
        line = line.split()
        if len(line) > 0:
            x_center_normalized = float(line[1])
            y_center_normalized = float(line[2])
            width_normalized = float(line[3])
            height_normalized = float(line[4])
            bbox = {}
            bbox['x'] = round(x_center_normalized *
                              width - (width_normalized * width / 2))
            bbox['y'] = round(y_center_normalized * height -
                              (height_normalized * height / 2))
            bbox['width'] = round(width_normalized * width)
            bbox['height'] = round(height_normalized * height)
            bbox['label'] = 'carrot' if line[0] == '3.0' else 'weed'
            bboxes.append(bbox)

    for bbox in bboxes:
        max_x = bbox['x'] + bbox['width']
        max_y = bbox['y'] + bbox['height']
        min_x = bbox['x']
        min_y = bbox['y']
        if max_x - min_x != max_y - min_y:
            if max_x - min_x > max_y - min_y:
                diff = max_x - min_x - max_y + min_y
                min_y -= diff / 2
                max_y += diff / 2
            else:
                diff = max_y - min_y - max_x + min_x
                min_x -= diff / 2
                max_x += diff / 2
        # Round and convert to int
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)

        if max_x - min_x != max_y - min_y:
            if max_x - min_x > max_y - min_y:
                max_x -= 1
            else:
                max_y -= 1
        bounding_boxes.append(
            {'label': bbox['label'], 'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y})
    return bounding_boxes


def removeOutOfBounds(annotations, img):
    """
    Remove bounding boxes that are out of the image size
    """
    # Get image size
    height, width, channels = img.shape
    # Remove bounding boxes that are out of the image size
    clean_bb = []
    for annotation in annotations:
        if annotation['x'] < 0 or annotation['y'] < 0 or annotation['x'] + annotation['width'] > width or annotation['y'] + annotation['height'] > height:
            continue
        clean_bb.append(annotation)
    return clean_bb


def calc_bbox_overlap(bbox1, bbox2):
    """
    Calculates the overlap between two bounding boxes
    """
    # Calculate overlap
    x1 = bbox1['x']
    y1 = bbox1['y']
    x2 = bbox1['x'] + bbox1['width']
    y2 = bbox1['y'] + bbox1['height']
    x3 = bbox2['x']
    y3 = bbox2['y']
    x4 = bbox2['x'] + bbox2['width']
    y4 = bbox2['y'] + bbox2['height']
    overlap_x = max(0, min(x2, x4) - max(x1, x3))
    overlap_y = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = overlap_x * overlap_y
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    overlap_ratio = overlap_area / (area1 + area2 - overlap_area)
    return overlap_ratio


def removeOverlap(annotations):
    """
    Filter out bounding boxes that overlap
    """
    overlapping_bounding_boxes = []
    for i in range(len(annotations)):
        for j in range(len(annotations)):
            if i != j:
                if calc_bbox_overlap(annotations[i], annotations[j]) > 0.0:
                    overlapping_bounding_boxes.append(annotations[i])
                    break
    for bbox in overlapping_bounding_boxes:
        annotations.remove(bbox)
    return annotations


def extract_bounding_boxes(annotations, img):
    """
    Extracts the bounding boxes from the image
    """
    # Extract bounding boxes from image
    cropped_imgs = []
    for annotation in annotations:
        cropped_img = img[annotation['y']:annotation['y'] + annotation['height'],
                          annotation['x']:annotation['x'] + annotation['width']]
        cropped_imgs.append({annotation['label']: cropped_img})
    return cropped_imgs


if __name__ == '__main__':
    main()
