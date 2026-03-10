"""
Communication functions that send the data to arduino.
Bounding Box center coordinates are transformed to global vehicle coordinates.
Coordinates and the class label are sent to the arduino as a string in the following format:
ID:<class_id> X:<x_coord> Y:<y_coord> Z:<z_coord>#
The # in the end indicates the end of the string.
"""

import numpy as np
import serial

# SERIAL CONNECTION
PORT = '/dev/ttyACM0'
conn = serial.Serial(
    port=PORT,
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1,
    xonxoff=False,
    rtscts=False,
    dsrdtr=False,
    write_timeout=2,
)


# TRANSFORMATION CONSTANTS
CAMERA_MATRIX = np.array([[315, 0, 320], [0, 315, 240], [0, 0, 1]])
TRANSFORMATION_VECTOR = np.array([[-0.147, -0.413, 0.169]])
ROTATION_MATRIX = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def transform_pixel_to_camera(img_coord: np.ndarray) -> np.ndarray:
    """
    Transform image coordinates to camera coordinates
    """
    img_coord = np.array(img_coord)
    img_coord = np.concatenate((img_coord, [1]))
    camera_coord = np.linalg.inv(CAMERA_MATRIX) @ img_coord

    camera_coord /= camera_coord[2]
    camera_coord *= 0.2
    return camera_coord


def transform_camera_to_world(camera_coord: np.ndarray) -> np.ndarray:
    """
    Transform camera coordinates to world coordinates
    """
    camera_coord = np.array(camera_coord)
    camera_coord = np.concatenate((camera_coord, [1]))
    transformation_matrix = np.concatenate(
        (ROTATION_MATRIX, TRANSFORMATION_VECTOR.T), axis=1)
    world_coord = transformation_matrix @ camera_coord
    return world_coord


def send_data_to_arduino(center_coord: np.ndarray, class_id: int):
    """
    Convert to vehicle coordinate system and send to arduino
    """
    world_coord = transform_camera_to_world(transform_pixel_to_camera(center_coord))
    # Round world to 4 decimals
    world_coord = np.round(world_coord, 4)
    data_string = f'ID:{class_id} X:{world_coord[0]} Y:{world_coord[1]} Z:{world_coord[2]}#'
    conn.write(data_string.encode())

def wait_for_start():
    """
    Wait for "START" message from arduino
    """
    while True:
        if conn.in_waiting > 0:
            data = conn.readline().decode()
            if data == 'START':
                return
def check_for_stop():
    """
    Check if "STOP" message is received from arduino
    """
    if conn.in_waiting > 0:
        data = conn.readline().decode()
        if data == 'STOP':
            return True
    return False