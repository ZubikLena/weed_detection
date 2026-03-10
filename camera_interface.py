"""
Interface class for image retrieval
"""

from abc import ABC, abstractmethod
import numpy as np

class CameraInterface(ABC):
    """
    Interface class for image retrieval
    """

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize the camera.
        """
        pass

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Returns the image from the camera as a numpy array
        """
        pass

    @abstractmethod
    def configure(self, config: dict):
        """
        Configure the camera with the given configuration
        For example, the resolution of the camera
        """
        pass