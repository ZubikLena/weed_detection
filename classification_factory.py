"""
Factory for creating classification objects
"""

from classification_interface import PlantClassificationInterface
from classification_leaf_contour import LeafContourClassification
from classification_cnn import CNNClassification

class ClassificationFactory(object):
    """
    Factory for creating classification objects.
    """

    @staticmethod
    def create_classifier(classification_type: str, config: dict) -> PlantClassificationInterface:
        """
        Create a classification object based on the given classification type.
        """
        if classification_type == "leaf_contour":
            return LeafContourClassification(config)
        elif classification_type == "cnn":
            return CNNClassification(config)
        else:
            raise ValueError("Unknown classification type: {}".format(classification_type))