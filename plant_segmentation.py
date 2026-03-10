import cv2
import numpy as np

from image_processing_interface import ImageProcessingInterface

class PlantSegmentation(ImageProcessingInterface):
    """
    Plant segmentation class
    """

    def __init__(self, config: dict):
        self.config = config

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image.
        """
        for processing_function, settings in self.config.items():
            image = getattr(self, processing_function)(image, settings)
        return image
    
    def green_amplifiy(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Amplify the green channel of the image
        """
        blue_channel = image[:, :, 0].astype(np.float32)
        green_channel = image[:, :, 1].astype(np.float32)
        red_channel = image[:, :, 2].astype(np.float32)
        amplified_green_image = 2 * green_channel - red_channel - blue_channel
        # Clamp between 0 and 255
        amplified_green_image = np.clip(amplified_green_image, 0, 255)
        # Convert back to uint8
        amplified_green_image = amplified_green_image.astype(np.uint8)
        return amplified_green_image

    def blur(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Blur the image.
        """
        return cv2.blur(image, (settings["kernel_size"], settings["kernel_size"]))

    def hsv_threshold(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Threshold the image with the given HSV settings and return as a binary image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array(settings["lower_hsv"])
        upper_hsv = np.array(settings["upper_hsv"])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        return mask
    
    def adaptive_threshold(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Threshold the image with the otus method
        """
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return mask

    def dilate(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Dilate the image
        """
        kernel = np.ones((settings["kernel_size"], settings["kernel_size"]), np.uint8)
        return cv2.dilate(image, kernel, iterations=settings["iterations"])

    def erode(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Erode the image
        """
        kernel = np.ones((settings["kernel_size"], settings["kernel_size"]), np.uint8)
        return cv2.erode(image, kernel, iterations=settings["iterations"])

    def denoise(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Denoise binary mask by erosion and dilation
        """
        kernel = np.ones((settings["kernel_size"], settings["kernel_size"]), np.uint8)
        return cv2.erode(cv2.dilate(image, kernel, iterations=settings["iterations"]), kernel, iterations=settings["iterations"])

    def configure(self, config: dict):
        """
        Configure the image processing with the given configuration
        """
        self.config = config
