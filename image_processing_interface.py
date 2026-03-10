"""
Image processing interface.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2

class ImageProcessingInterface(ABC):
    """
    Interface class for image processing.
    """

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize the image processing.
        """
        pass

    @abstractmethod
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image.
        """
        pass

    @abstractmethod
    def configure(self, config: dict):
        """
        Configure the image processing with the given configuration
        For example decide which processing steps to use with which settings
        """
        pass