"""
Factory for creating image processing objects
"""

from image_processing_interface import ImageProcessingInterface
from plant_segmentation import PlantSegmentation

class ImageProcessingFactory(object):
    """
    Factory for creating image processing objects.
    """

    @staticmethod
    def create_image_processing(image_processing_type: str, config: dict) -> ImageProcessingInterface:
        """
        Create a image processing object based on the given image processing type.
        """
        if image_processing_type == "plant_segmentation":
            return PlantSegmentation(config)
        else:
            raise ValueError("Unknown image processing type: {}".format(image_processing_type))