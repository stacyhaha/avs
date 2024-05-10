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
from flask import Flask, request, jsonify

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from localizer import Localizer
from utils import decompress_base64_to_image


app = Flask(__name__)
CORS(app)


logger = logging.getLogger()
fh = logging.FileHandler("test_localizer.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)  # 把formater绑定到fh上
logger.addHandler(fh)
logger.warning("test")  


def run_flask(shared_data):
    @app.route("/backend/img", methods=["POST"])
    def get_img():
        logger.info("get img")
        img = request.json["img"]
        shared_data.value = img.encode()
        return {"status": "ok"}
    app.run(host="0.0.0.0", port=930)


def test_basic_driver(shared_data):
    localizer = Localizer()
    while True:
        if len(shared_data.value) > 0:
            img = shared_data.value.decode()
            img = decompress_base64_to_image(img)
            
            st = time.time()
            pos = localizer.localize(img)
            logger.info(f"localizer process one pic {time.time()-st:.3f} s")
            logger.info(str(pos))

            img = np.array(img)
            opencv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Image", opencv_image)
            cv2.waitKey(200)
            



if __name__ == "__main__":
    img_str = multiprocessing.Array("c", 10*1024*1024, lock=False)
    app_process = multiprocessing.Process(target=run_flask, args=(img_str, ))
    test_basic_driver_process = multiprocessing.Process(target=test_basic_driver, args=(img_str, ))

    test_basic_driver_process.start()
    app_process.start()
    
    app_process.join()
    test_basic_driver_process.join()




