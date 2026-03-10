from __future__ import annotations
import os
from typing import overload
import numpy as np
import yaml
import cv2
BASE_DIR = "./data/yaml/annotations"
IMG_DIR = "./data/yaml/images"
OUTPUT_DIR = "./data/yaml/soil"


def main():

    # Find all YAML files in the directory
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.yaml')]

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in files:
        print(file)
        print('\n')
        print('\n')
        print('\n')
        annotations = getBoundingBoxes(file)
        dirtBoundingBoxes = getDirtBoundingBoxes(annotations, file)
        if dirtBoundingBoxes and len(dirtBoundingBoxes) > 0:
            cropped_imgs = extract_bounding_boxes(dirtBoundingBoxes, file)
            if cropped_imgs:
                for img in cropped_imgs:
                    for key, value in img.items():
                        # Check if output directory exists
                        if not os.path.exists(OUTPUT_DIR):
                            os.makedirs(OUTPUT_DIR)
                        # Save image with rising index
                        cv2.imwrite(os.path.join(OUTPUT_DIR, file.replace('annotation.yaml', '') + str(
                            len(os.listdir(OUTPUT_DIR))) + '.png'), value)


def getDirtBoundingBoxes(annotations, file):
    # [{'label': 'weed', 'x': 810, 'y': 85, 'width': 365, 'height': 365}, {'label': 'weed', 'x': 637, 'y': 395, 'width': 433, 'height': 433}, {'label': 'crop', 'x': 1095, 'y': 869, 'width': 82, 'height': 82}, {'label': 'weed', 'x': 708, 'y': 718, 'width': 217, 'height': 217}, {'label': 'weed', 'x': 533, 'y': 632, 'width': 245, 'height': 245}, {'label': 'weed', 'x': 329, 'y': 569, 'width': 212, 'height': 212}, {'label': 'weed', 'x': 316, 'y': 580, 'width': 159, 'height': 159}, {'label': 'weed', 'x': 455, 'y': 755, 'width': 138, 'height': 138}, {'label': 'weed', 'x': 643, 'y': 261, 'width': 119, 'height': 119}, {'label': 'crop', 'x': 705, 'y': 233, 'width': 229, 'height': 229}, {'label': 'crop', 'x': 669, 'y': -26, 'width': 270, 'height': 270}, {'label': 'weed', 'x': 848, 'y': 1, 'width': 146, 'height': 146}, {'label': 'weed', 'x': 492, 'y': 514, 'width': 108, 'height': 108}, {'label': 'weed', 'x': 276, 'y': 279, 'width': 290, 'height': 290}, {'label': 'weed', 'x': 274, 'y': 169, 'width': 265, 'height': 265}, {'label': 'weed', 'x': 397, 'y': 89, 'width': 394, 'height': 394}, {'label': 'weed', 'x': 561, 'y': 481, 'width': 124, 'height': 124}]

    # [label, x, y, width, height] = annotation
    if os.path.exists(os.path.join(
            IMG_DIR, file.replace('annotation.yaml', 'image.png'))):
        img = cv2.imread(os.path.join(
            IMG_DIR, file.replace('annotation.yaml', 'image.png')))
        print(img.shape)
        height, width, channels = img.shape
        dirtBoxes = []
        coordinateStepSize = 10
        radiusCutoff = 150
        radiusStepSize = 15
        for x in range(1, height, coordinateStepSize):
            if len(dirtBoxes) < 4:
                for y in range(1, width, coordinateStepSize):
                    if len(dirtBoxes) < 4:
                        s = radiusCutoff  # side length of square
                        lastBox = {'x': x-s, 'y': y-s, 'width': 2 *
                                   s + 1, 'height': 2*s + 1, 'label': 'soil'}
                        # get biggest possible circle that does not overlap with another bounding box or the edge of the image
                        # print(noOverlapBB)
                        # print(noOverlapDB)
                        # print(inImage)
                        # print('---')
                        while squareNotOverlapping(lastBox, annotations) and squareNotOverlapping(lastBox, dirtBoxes) and squareInsideImage(x, y, s, img):
                            lastBox = {'x': x-s, 'y': y-s, 'width': 2 *
                                       s + 1, 'height': 2*s + 1, 'label': 'soil'}
                            s = s + radiusStepSize
                        if s > radiusCutoff:
                            dirtBoxes.append(lastBox)
        print('returning dirtboxes')
        print(dirtBoxes)
        return dirtBoxes


def squareNotOverlapping(dirtBox, otherBoxes):
    noOverlap = True
    for box in otherBoxes:
        noOverlap = noOverlap & (calc_bbox_overlap(box, dirtBox) == 0.0)
    return noOverlap


def squareInsideImage(x, y, s, img):
    """
    Remove bounding boxes that are out of the image size
    """
    square = {'x': x-s, 'y': y-s, 'width': 2*s + 1, 'height': 2*s + 1}
    height, width, channels = img.shape
    return not (square['x'] < 0 or square['y'] < 0 or square['x'] + square['width'] > width or square['y'] + square['height'] > height)


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


def getBoundingBoxes(file):
    """
    Reads the YAML file and returns a list of bounding boxes and their labels
    """
    with open(os.path.join(BASE_DIR, file), 'r') as stream:
        try:
            data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
            # Find bounding box in annotation contour (points)
            bounding_boxes = []
            for annotation in data_loaded['annotation']:
                points = annotation['points']
                x = points['x']
                y = points['y']
                # Calculate bounding box from min and max of x and y
                # Check if x or y are lists
                if not isinstance(x, list) or not isinstance(y, list):
                    continue

                min_x = min(x)
                max_x = max(x)
                min_y = min(y)
                max_y = max(y)
                # Increase edge size if bounding box not square
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
                # Append dict of bounding box and label to list
                bounding_boxes.append(
                    {'label': annotation['type'], 'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y})
            return bounding_boxes
        except yaml.YAMLError as exc:
            print(exc)


def extract_bounding_boxes(annotations, file):
    """
    Extracts the bounding boxes from the image
    """
    # Read image
    pathToImage = os.path.join(
        IMG_DIR, file.replace('annotation.yaml', 'image.png'))
    if os.path.exists(pathToImage):
        img = cv2.imread(pathToImage)
        # Extract bounding boxes from image
        cropped_imgs = []
        for annotation in annotations:
            cropped_img = img[annotation['y']:annotation['y'] + annotation['height'],
                              annotation['x']:annotation['x'] + annotation['width']]
            cropped_imgs.append({annotation['label']: cropped_img})
        return cropped_imgs


if __name__ == '__main__':
    main()
