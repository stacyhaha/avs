import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import requests
from PIL import Image


def ros_image_to_cv2(ros_image):
    """Convert ROS image to OpenCV format."""
    frame = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_image


def connected_components(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return cv2.connectedComponentsWithStats(binary_image, connectivity=8)


def find_path(corrected_image, RunwayNum):
    return np.where(corrected_image == RunwayNum, 255, 0).astype(np.uint8)


def calculate_driving_commands(curvature):
    # This method should be developed based on your specific vehicle's dynamics.
    # Placeholder logic: More curvature means slower speed and more direction adjustment.
    speed = max(0.1, 1 - curvature)  # Example placeholder logic
    direction = curvature * 30  # Example placeholder logic
    bias = curvature * 5  # Example placeholder logic
    return speed, direction, bias


def calculate_curvature(path_image):
    edges = cv2.Canny(path_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]
        return np.std(angles)
    return 0


def trapezoidal_correction(labels, MapJ):
    corrected_image = np.zeros_like(labels)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            corrected_x = MapJ[y, x]
            if corrected_x < 255:
                corrected_image[y, x] = labels[y, corrected_x]
            else:
                corrected_image[y, x] = 0
    return corrected_image


class BasicDriver:
    def __init__(self, runway_number=2):
        self.runway_number = runway_number

    def drive(self, ros_image: Image):
        """
        Analyze the provided image and return driving commands.

        Args:
        ros_image (sensor_msgs.msg.Image): The image from ROS camera.

        Returns:
        tuple: Driving commands (speed, direction, bias)
        """
        # Convert ROS image to OpenCV format
        image = ros_image_to_cv2(ros_image)

        # Process the image to find the path curvature
        curvature, path_image = self.process_image(image, self.runway_number)

        # Determine the driving commands based on curvature
        speed, direction, bias = calculate_driving_commands(curvature)

        return (speed, direction, bias)

    def process_image(self, image, RunwayNum):
        num_labels, labels, stats, centroids = connected_components(image)
        height, width = labels.shape
        MapJ = self.generate_mapj(height, width)
        corrected_labels = trapezoidal_correction(labels, MapJ)
        path_image = find_path(corrected_labels, RunwayNum)
        curvature = calculate_curvature(path_image)
        return curvature, path_image


    def generate_mapj(self, height, width):
        """
        Generate a mapping matrix for trapezoidal correction based on image dimensions.

        Args:
        height (int): The height of the image.
        width (int): The width of the image.

        Returns:
        numpy.ndarray: A 2D array where each element is the corrected x-coordinate for that pixel.
        """
        MapJ = np.zeros((height, width), dtype=int)
        for y in range(height):
            for x in range(width):
                # 这里假设每行的偏移量是基于行号的正弦函数，你可以根据需要调整这个函数
                shift = int(10 * np.sin(np.pi * y / height))
                # 确保偏移后的x坐标不会超出图像边界
                corrected_x = np.clip(x + shift, 0, width - 1)
                MapJ[y, x] = corrected_x
        return MapJ
