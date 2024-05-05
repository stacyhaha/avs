import cv2
import numpy as np
from PIL import Image

def save_image(img_array, file_name):
    """Save an image from a numpy array."""
    save_path = f"D:\\ISY5003\\ISY5003\\avs\\output\\{file_name}.png"
    cv2.imwrite(save_path, img_array)

def calculate_curvature(lines):
    """Calculate curvature based on the detected lines."""
    if lines is None:
        return 0
    lengths = []
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        lengths.append(length)
        angles.append(angle)
    angles = np.rad2deg(angles) % 360
    angles = np.unwrap(np.deg2rad(angles))
    if len(lengths) > 1:
        weighted_angle_change = np.sum(np.diff(angles) * lengths[:-1]) / np.sum(lengths[:-1])
    else:
        weighted_angle_change = 0
    return np.abs(weighted_angle_change) * 6

def calculate_driving_commands(curvature):
    """Calculate driving commands based on curvature."""
    if curvature > 0.1:
        speed = 45
        direction = 90
        bias =0
    else:
        speed = 30
        direction = 90
        bias = -30
    return speed, direction, bias

class BasicDriver:
    def __init__(self, runway_number=2):
        self.runway_number = runway_number

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS image to OpenCV format and apply masking to the top 2/3 of the image."""
        numpy_image = np.array(ros_image)
        frame = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        height = frame.shape[0]
        top_mask_height = int(2 / 3 * height)
        frame[:top_mask_height, :] = 0  # Setting top 2/3 of the image to black
        return frame

    def drive(self, ros_image: Image):
        """Analyze the provided image and return driving commands."""
        image = self.ros_image_to_cv2(ros_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        curvature = calculate_curvature(lines) if lines is not None else 0
        return calculate_driving_commands(curvature)
