"""
Interface class for classification of plants
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

class PlantClassificationInterface(ABC):
    """
    Interface class for classification of plants
    """

    @abstractmethod

    def __init__(self, config: dict):
        """
        Initialize the plant classification.
        """
        pass

    @abstractmethod
    def classify_plants(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Classify the plants in the image and return a dictionary with the labels and bounding boxes.
        {"labels": [], "bboxes": []}
        """
        pass

    def cluster_contours(self, contours: list, distance_thresh: float) -> dict:
        """
        Cluster the contours into groups by distance
        """
        close_indices = []
        # Calculate the distance between each contour
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                distance = self.calculate_distance(contours[i], contours[j])
                if distance < distance_thresh:
                    close_indices.append([i, j])
        G1 = nx.Graph()
        G1.add_edges_from(close_indices)
        clusters = nx.connected_components(G1)
        clusters = [list(cluster) for cluster in clusters]
        # Add not clustered contours to the clusters
        for i in range(len(contours)):
            if i not in [index for cluster in clusters for index in cluster]:
                clusters.append([i])
        # Concetenate clustered contours
        clustered_contours = [np.concatenate(
            [contours[index] for index in cluster]) for cluster in clusters]
        return clustered_contours

    @staticmethod
    def calculate_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Iterate over all points in a contour and find minimum distance between the points
        """
        distance = cdist(contour1[:, 0, :],
                         contour2[:, 0, :], metric="euclidean")
        return np.min(distance)