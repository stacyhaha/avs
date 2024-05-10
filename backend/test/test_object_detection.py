import cv2
import os
import sys
import time
import logging
import numpy as np
from PIL import Image
import multiprocessing
from flask_cors import CORS
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from object_detection.predict import ObjectDetection
from utils import decompress_base64_to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
fh = logging.FileHandler("test_detection.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)  # 把formater绑定到fh上
logger.addHandler(fh)
logger.warning("test")  

app = Flask(__name__)
CORS(app)

def run_flask(shared_data):
    @app.route("/backend/img", methods=["POST"])
    def get_img():
        logger.info("get img")
        img = request.json["img"]
        shared_data.value = img.encode()
        return {"status": "ok"}
    app.run(host="0.0.0.0", port=930)


def test_detection(shared_data):
    logger.info("upload detection model")
    tf_checkpoint = 'object_detection/object_detection_part/traffic_light/weights/best.pt'
    ts_checkpoint = 'object_detection/object_detection_part/traffic_signs/weights/best.pt'
    object_detector = ObjectDetection(tf_checkpoint, ts_checkpoint)
    logger.info("loaded detection model successfully")
    i = 0
    while True:
        logger.info(f"shared_data length {len(shared_data.value)}")
        if len(shared_data.value) > 0:
            logger.info("basic drive get pic")
            img = shared_data.value.decode()
            img = decompress_base64_to_image(img)
            
            img = np.array(img)
            #opencv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Image", opencv_image)
            # cv2.waitKey(100)

            st = time.time()
            try:
                res = object_detector.detect(img)
                logger.info(f"detect one pic {time.time()-st:.3f} s")
            except Exception as e:
                logger.error(e)
            # try:
            #     request_res = requests.post("http://172.17.0.1:9831/edge/cmd", json={"cmd":cmd})
            # except:
            #     pass
            try:
                detect_image = res[-1]
                opencv_image = cv2.cvtColor(detect_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("detect_image", opencv_image)
                cv2.waitKey(10)
            except Exception as e:
                logger.error(e)
            # try:
            #     logger.info("saving")
            #     img = np.array(img)
            #     
            #     cv2.imwrite(f"{i}.png", opencv_image)
            #     logger.info(i)
            #     if i == 0:
            #         with open("cmd.txt", "w") as f:
            #             f.write("")
            #     with open("cmd.txt", "a") as f:
            #         f.write(f"{i}.png, cmd {str(cmd)}\n")
            #     i += 1
            #     
            # except Exception as e:
            #     logger.error(str(e))


        
if __name__ == "__main__":
    img_str = multiprocessing.Array("c", 10*1024*1024, lock=False)
    app_process = multiprocessing.Process(target=run_flask, args=(img_str, ))
    test_detect_process = multiprocessing.Process(target=test_detection, args=(img_str, ))

    test_detect_process.start()
    app_process.start()
    
    app_process.join()
    test_detect_process.join()