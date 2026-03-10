from __future__ import annotations
import os
import numpy as np
import yaml
import cv2
BASE_DIR = "./data/yaml/annotations"
IMG_DIR = "./data/yaml/images"
OUTPUT_DIR = "./data/yaml/images_cropped"


def main():

    # Find all YAML files in the directory
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.yaml')]

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in files:
        annotations = getBoundingBoxes(file)
        # draw_bounding_boxes(annotations, file)
        annotations = removeOutOfBounds(annotations, file)
        annotations = removeOverlap(annotations)
        if len(annotations) > 0:
            cropped_imgs = extract_bounding_boxes(annotations, file)
            if cropped_imgs:
                for img in cropped_imgs:
                    for key, value in img.items():
                        # Check if output directory exists
                        if not os.path.exists(os.path.join(OUTPUT_DIR, key)):
                            os.makedirs(os.path.join(OUTPUT_DIR, key))
                        # Save image with rising index
                        cv2.imwrite(os.path.join(OUTPUT_DIR, key, str(
                            len(os.listdir(os.path.join(OUTPUT_DIR, key)))) + '.png'), value)


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


def removeOutOfBounds(annotations, file):
    """
    Remove bounding boxes that are out of the image size
    """
    # Read image
    img = cv2.imread(os.path.join(BASE_DIR, file.replace('.yaml', '.png')))
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


def draw_bounding_boxes(annotations, file):
    """
    Draws the bounding boxes on the image
    """
    # Read image
    img = cv2.imread(os.path.join(
        IMG_DIR, file.replace('annotation.yaml', 'image.png')))
    # Draw bounding boxes on image
    for annotation in annotations:
        cv2.rectangle(img, (annotation['x'], annotation['y']), (annotation['x'] +
                      annotation['width'], annotation['y'] + annotation['height']), (0, 0, 255), 2)
        cv2.putText(img, annotation['label'], (annotation['x'],
                    annotation['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Save image
    cv2.imwrite(os.path.join(OUTPUT_DIR, file.replace(
        'annotation.yaml', 'image.png')), img)


if __name__ == '__main__':
    main()
