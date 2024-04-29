import rospy
import base64
import requests
import numpy as np
import logging
import sys
import cv2
from io import BytesIO
from sensor_msgs.msg import Image

# 采集图片，使用base64压缩后，发到backend

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

backend_url = "http://localhost:930"
url_img = f"{backend_url}/backend/img"

def compress_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    base64_encoded = base64.b64encode(buffer).decode("utf-8")
    return base64_encoded

def image_callback(ros_image):
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    image_base64 = compress_image_to_base64(image)
    res = requests.post(url_img, json={"img": image_base64})
    logger.info(res.status_code)


def col_img():
    rospy.init_node("collect_data", anonymous=True)
    img_pub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback)



