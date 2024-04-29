import requests
import cv2
from io import BytesIO
import base64
import time
import numpy as np


img_url = "http://localhost:930/img_test"

def compress_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    base64_encoded = base64.b64encode(buffer).decode("utf-8")
    return base64_encoded


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        image_array = np.array(frame)
        if not ret:
            continue

        compressed_img = compress_image_to_base64(image_array)
        try:
            requests.post(img_url, json={"img": compressed_img})
        except:
            continue
        cv2.waitKey(3000)

if __name__ == "__main__":
    main()