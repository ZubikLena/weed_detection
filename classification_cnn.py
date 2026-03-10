import cv2
import numpy as np
from classification_interface import PlantClassificationInterface
import tensorflow as tf


class CNNClassification(PlantClassificationInterface):
    """
    Classify the plants in the image with a CNN
    """

    def __init__(self, config: dict):
        """
        Initialize the plant classification.
        """
        self.config = config
        self.model = tf.keras.models.load_model(config["model_path"])

    def classify_plants(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Classify the plants in the image.
        """
        # Get the contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Cluster the contours
        contours = self.cluster_contours(
            contours, self.config["cluster_distance_threshold"])
        # Get the leaf area of each contour
        areas = [cv2.contourArea(contour) for contour in contours]
        # Get the bounding boxes of each contour
        bboxes = [cv2.boundingRect(contour) for contour in contours]

        # Remove small contours
        min_leaf_area = self.config["min_leaf_area"]
        contours = [contour for contour, area in zip(
            contours, areas) if area > min_leaf_area]
        bboxes = [bbox for bbox, area in zip(
            bboxes, areas) if area > min_leaf_area]

        # Remove bounding boxes that touch left or right border
        bboxes = [bbox for bbox in bboxes if bbox[0] >
                  15 and bbox[0] + bbox[2] < (image.shape[1]-15)]

        cropped_images = self.rectangle_crop(image, bboxes)

        # Resize images to (224, 224)
        resized_images = [cv2.resize(image, (112, 112))
                          for image in cropped_images]

        # Convert images to tensor
        images = np.array(resized_images)
        images = images.astype(np.float32)

        if images.shape[0] == 0:
            return {"labels": [], "bboxes": []}
        # Predict the labels
        labels = self.model.predict(images)
        labels = np.argmax(labels, axis=1)

        # Return the labels and bounding boxes as a dict
        return {"labels": labels, "bboxes": bboxes}

    def rectangle_crop(self, image: np.ndarray, bboxes: list) -> list:
        bb_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                      for bbox in bboxes]
        bb_sizes = [max(bbox[2], bbox[3]) for bbox in bboxes]

        cropped_images = []

        for bb_center, bb_size in zip(bb_centers, bb_sizes):
            if bb_center[1] - bb_size / 2 < 0:
                x_tl = 0
                x_br = bb_size
            elif bb_center[1] + bb_size / 2 > image.shape[0]:
                x_tl = image.shape[0] - bb_size
                x_br = image.shape[0]
            else:
                x_tl = bb_center[1] - bb_size / 2
                x_br = bb_center[1] + bb_size / 2
            if bb_center[0] - bb_size / 2 < 0:
                y_tl = 0
                y_br = bb_size
            elif bb_center[0] + bb_size / 2 > image.shape[1]:
                y_tl = image.shape[1] - bb_size
                y_br = image.shape[1]
            else:
                y_tl = bb_center[0] - bb_size / 2
                y_br = bb_center[0] + bb_size / 2
            cropped_image = image[int(x_tl):int(x_br), int(y_tl):int(y_br)]
            cropped_images.append(cropped_image)
        return cropped_images

    def calc_distance(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Calculate the distance between two contours with fourier descriptors
        """
        _, similarity = self.cf.estimateTransformation(contour1, contour2)
        return similarity
