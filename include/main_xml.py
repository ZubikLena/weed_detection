from __future__ import annotations
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
BASE_DIR = "./data/xml/annotations"
IMG_DIR = "./data/xml/raw images"
OUTPUT_DIR = "./data/xml/images_cropped"

"""
XML EXAMPLE:
<annotation>
	<filename>32529.jpg</filename>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<object>
		<name>weed</name>
		<bndbox>
			<xmin>1050</xmin>
			<ymin>28</ymin>
			<xmax>1123</xmax>
			<ymax>65</ymax>
		</bndbox>
	</object>
</annotation>
"""


def main():

    # Find all xml files in the directory
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.xml')]

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


def removeOutOfBounds(annotations, file):
    """
    Remove bounding boxes that are out of the image size
    """
    # Read image
    img = cv2.imread(os.path.join(IMG_DIR, file.replace('.xml', '.jpg')))
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
    img = cv2.imread(os.path.join(IMG_DIR, file.replace('.xml', '.jpg')))
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
    img = cv2.imread(os.path.join(IMG_DIR, file.replace('.xml', '.jpg')))
    # Draw bounding boxes on image
    for annotation in annotations:
        cv2.rectangle(img, (annotation['x'], annotation['y']), (annotation['x'] +
                      annotation['width'], annotation['y'] + annotation['height']), (0, 0, 255), 2)
        cv2.putText(img, annotation['label'], (annotation['x'],
                    annotation['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Save image
    cv2.imwrite(os.path.join(OUTPUT_DIR, file.replace('xml', 'jpg')), img)


if __name__ == '__main__':
    main()
