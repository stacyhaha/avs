import cv2
import numpy as np
from PIL import Image

def save_image(img_array, file_name):
    """Save an image from a numpy array."""
    save_path = f"D:\\ISY5003\\ISY5003\\avs\\output\\{file_name}.png"
    cv2.imwrite(save_path, img_array)


def calculate_curvature(lines):
    if lines is None:
        return 0

    lengths = []
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # Ensure the line is reshaped to flat array if not already
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        lengths.append(length)
        angles.append(angle)

    # Normalize angles
    angles = np.rad2deg(angles) % 360
    angles = np.unwrap(np.deg2rad(angles))  # Unwrap angles to handle periodicity

    # Calculate the weighted average of angle changes
    if len(lengths) > 1:
        weighted_angle_change = np.sum(np.diff(angles) * lengths[:-1]) / np.sum(lengths[:-1])
    else:
        weighted_angle_change = 0

    return np.abs(weighted_angle_change) * 6  # Curvature as the absolute value of angle change


def calculate_driving_commands(curvature):
    """Calculate driving commands based on curvature."""
    if curvature > 0.1:
        speed = 45
        direction = 90
        bias = int(np.clip(curvature * 5, -30, 30))
    else:
        speed = int(max(10, 60 - 50 * curvature))
        direction = 90
        bias = int(np.clip(curvature * -600, -30, 30))
    return speed, direction, bias

class BasicDriver:
    def __init__(self, runway_number=2):
        self.runway_number = runway_number

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS image to OpenCV format and apply masking to top and bottom thirds."""
        numpy_image = np.array(ros_image)
        frame = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # Mask the top 1/3 of the image
        height = frame.shape[0]
        top_mask_height = int(1 / 3 * height)
        frame[:top_mask_height, :] = 0  # Setting top 1/3 of the image to black

        # Mask the bottom 1/3 of the image
        bottom_mask_start = int(2 / 3 * height)
        frame[bottom_mask_start:, :] = 0  # Setting bottom 1/3 of the image to black

        return frame

    def apply_trapezoidal_correction(self, image):
        """Apply trapezoidal correction to the image."""
        height, width = image.shape[:2]
        src_points = np.float32([
            [width * 0.13, height * (1/3)],  # Near top of middle third
            [width * 0.9, height * (1/3)],  # Near top of middle third
            [width * 0.01, height * (2/3)],  # Near bottom of middle third
            [width * 0.9, height * (2/3)]    # Near bottom of middle third
        ])
        dst_points = np.float32([
            [0, 0],                        # Top left
            [width, 0],                    # Top right
            [0, height],                   # Bottom left
            [width, height]                # Bottom right
        ])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected_image = cv2.warpPerspective(image, M, (width, height))
        save_image(corrected_image, "trapezoidal_corrected_image")
        return corrected_image

    def drive(self, ros_image: Image):
        """Analyze the provided image and return driving commands."""
        image = self.ros_image_to_cv2(ros_image)
        corrected_image = self.apply_trapezoidal_correction(image)
        save_image(corrected_image, "corrected_image")
        gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            curvature = calculate_curvature(lines)
        else:
            curvature = 0
        return calculate_driving_commands(curvature)


# if __name__ == "__main__":
#     basic_driver = BasicDriver()
#     image_path = r"D:\ISY5003\ISY5003\avs\data\11.png"
#     image = Image.open(image_path)
#     res = basic_driver.drive(image)
#     print(res)
