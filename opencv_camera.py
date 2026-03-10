import cv2
import numpy as np
import glob
import os
from camera_interface import CameraInterface


class Webcam(CameraInterface):
    """
    Opencv webcam class
    """

    def __init__(self, config: dict):
        self.config = config
        self.camera = cv2.VideoCapture(0)
        self.configure(config)

    def get_image(self) -> np.ndarray:
        """
        Returns the image from the camera as a numpy array
        """
        _, image = self.camera.read()
        return image

    def configure(self, config: dict):
        """
        Configure the camera with the given configuration
        For example, the resolution of the camera
        """
        if "width" in config and "height" in config:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        if "fps" in config:
            self.camera.set(cv2.CAP_PROP_FPS, config["fps"])


class FileCamera(CameraInterface):
    """
    Opencv file camera class
    """

    def __init__(self, config: dict):
        self.config = config
        if "file" in config:
            self.camera = cv2.VideoCapture(config["file"])

        elif "folder" in config:
            types = ('*.png', '*.jpg', '*.JPG')  # the tuple of file types
            self.image_paths = []
            for files in types:
                self.image_paths.extend(
                    glob.glob(config["folder"] + os.sep + files))
            self.image_paths.sort()
            self.image_index = 0

    def get_image(self) -> np.ndarray:
        """
        Returns the image from the camera as a numpy array
        """
        if "file" in self.config:
            _, image = self.camera.read()
        elif "folder" in self.config:
            if self.image_index >= len(self.image_paths):
                image = None
            else:
                image = cv2.imread(self.image_paths[self.image_index])
            self.image_index += 1
        if "width" in self.config and "height" in self.config:
            image = cv2.resize(
                image, (self.config["width"], self.config["height"]))

        # Crop image
        # THIS IS ONLY DONE BECAUSE IMAGES ARE FAULTY ON THE RIGHT SIDE
        if image is not None:
            image = image[:, :int(image.shape[1]/5*4)]
            
        return image

    def configure(self, config: dict):
        """
        Configure the camera with the given configuration
        For example, the resolution of the camera
        """
        self.config = config
