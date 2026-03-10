import cv2
import numpy as np
from classification_interface import PlantClassificationInterface
from cv2.ximgproc import createContourFitting


class LeafContourClassification(PlantClassificationInterface):
    """
    Classify the plants in the image by their leaf area
    """

    def __init__(self, config: dict):
        """
        Initialize the plant classification.
        """
        self.config = config
        self.cf = createContourFitting(100)
        self.carrot_cnts = []
        self.weed_cnts = []

    def create_contour_dataset(self, carrot_masks: list, weed_masks: list):
        """
        Extract contours from images and safe as a dataset
        """
        for carrot_mask in carrot_masks:
            # Get the contours
            contours, _ = cv2.findContours(
                carrot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Cluster the contours
            contours = self.cluster_contours(
                contours, self.config["cluster_distance_threshold"])
            # Get the leaf area of each contour
            areas = [cv2.contourArea(contour) for contour in contours]

            # Remove small contours
            min_leaf_area = self.config["min_leaf_area"]
            contours = [contour for contour, area in zip(
                contours, areas) if area > min_leaf_area]
            if len(contours) > 0:
                self.carrot_cnts.append(contours[0])

        for weed_mask in weed_masks:
            # Get the contours
            contours, _ = cv2.findContours(
                weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Cluster the contours
            contours = self.cluster_contours(
                contours, self.config["cluster_distance_threshold"])
            # Get the leaf area of each contour
            areas = [cv2.contourArea(contour) for contour in contours]
            # Remove small contours
            min_leaf_area = self.config["min_leaf_area"]
            contours = [contour for contour, area in zip(
                contours, areas) if area > min_leaf_area]
            if len(contours) > 0:
                self.weed_cnts.append(contours[0])

        self.carrot_cnts = self.carrot_cnts[:self.config["max_carrot_cnts"]]
        self.weed_cnts = self.weed_cnts[:self.config["max_weed_cnts"]]

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
        areas = [area for area in areas if area > min_leaf_area]

        labels = []
        for contour in contours:
            carrot_dists = []
            for carrot_cnts in self.carrot_cnts:
                carrot_dists.append(self.calc_distance(contour, carrot_cnts))
            weed_dists = []
            for weed_cnts in self.weed_cnts:
                weed_dists.append(self.calc_distance(contour, weed_cnts))
            if np.min(carrot_dists) < np.min(weed_dists):
                labels.append(0)
            else:
                labels.append(1)
        # Return the labels and bounding boxes as a dict
        return {"labels": labels, "bboxes": bboxes}

    def calc_distance(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Calculate the distance between two contours with fourier descriptors
        """
        _, similarity = self.cf.estimateTransformation(contour1, contour2)
        return similarity

    def get_contour_similarity(self, contours: list) -> np.ndarray:
        """
        Calculate the distance matrix between the contours
        """
        # Calculate the distance matrix between the descriptors
        distance_matrix = np.zeros((len(contours), len(contours)))
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                distance = self.calc_distance(contours[i], contours[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix
