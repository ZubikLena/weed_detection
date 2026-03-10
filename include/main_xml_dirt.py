from __future__ import annotations
import os
from typing import overload
import numpy as np
import cv2
import xml.etree.ElementTree as ET

BASE_DIR = "./data/xml_soil/annotations"
IMG_DIR = "./data/xml_soil/raw images"
OUTPUT_DIR = "./data/xml_soil/soil"


def main():

    # Find all YAML files in the directory
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.xml')]

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
                for index, img in enumerate(cropped_imgs):
                    for key, value in img.items():
                        # Check if output directory exists
                        if not os.path.exists(OUTPUT_DIR):
                            os.makedirs(OUTPUT_DIR)
                        # Save image with rising index
                        cv2.imwrite(os.path.join(OUTPUT_DIR, file.replace('.xml', '_') + str(
                            index) + '.jpg'), value)


def getDirtBoundingBoxes(annotations, file):
    print(annotations)

    # [label, x, y, width, height] = annotation
    if os.path.exists(os.path.join(
            IMG_DIR, file.replace('.xml', '.jpg'))):
        img = cv2.imread(os.path.join(
            IMG_DIR, file.replace('.xml', '.jpg')))
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
    Reads the XML file and returns a list of bounding boxes and their labels
    """
    # Read XML file
    tree = ET.parse(os.path.join(BASE_DIR, file))
    root = tree.getroot()
    # Get bounding boxes and labels
    bounding_boxes = []
    bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == 'name':
                    label = grandchild.text
                elif grandchild.tag == 'bndbox':
                    for greatgrandchild in grandchild:
                        if greatgrandchild.tag == 'xmin':
                            x = int(greatgrandchild.text)
                        elif greatgrandchild.tag == 'ymin':
                            y = int(greatgrandchild.text)
                        elif greatgrandchild.tag == 'xmax':
                            width = int(greatgrandchild.text) - x
                        elif greatgrandchild.tag == 'ymax':
                            height = int(greatgrandchild.text) - y
                    bboxes.append({'x': x, 'y': y, 'width': width,
                                  'height': height, 'label': label})
                    # Increase edge size if bounding box not square
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


def extract_bounding_boxes(annotations, file):
    """
    Extracts the bounding boxes from the image
    """
    # Read image
    pathToImage = os.path.join(
        IMG_DIR, file.replace('.xml', '.jpg'))
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
