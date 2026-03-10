"""
Factory for creating camera objects.
"""
from camera_interface import CameraInterface
from opencv_camera import FileCamera, Webcam

class CameraFactory(object):
    """
    Factory for creating camera objects.
    """

    @staticmethod
    def create_camera(camera_type: str, config: dict) -> CameraInterface:
        """
        Create a camera object based on the given camera type.
        """
        if camera_type == "webcam":
            return Webcam(config)
        elif camera_type == "filecam":
            return FileCamera(config)
        else:
            raise ValueError("Unknown camera type: {}".format(camera_type))