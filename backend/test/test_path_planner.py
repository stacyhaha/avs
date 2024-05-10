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

from path_planner import exe
from utils import decompress_base64_to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
fh = logging.FileHandler("test_path.log")
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


def test_basic_driver(shared_data):
    logger.info("upload basic driver")
    start_pos = (20, 20)
    end_pos = (180, 220)
    res = exe(start_pos, end_pos)
    logger.info(str(res))

    try:
        res = requests.post("http://172.17.0.1:9831/edge/cmd", json={"cmd":res})
        logger.info("post")
    except:
        pass
            

if __name__ == "__main__":
    img_str = multiprocessing.Array("c", 10*1024*1024, lock=False)
    app_process = multiprocessing.Process(target=run_flask, args=(img_str, ))
    test_basic_driver_process = multiprocessing.Process(target=test_basic_driver, args=(img_str, ))

    test_basic_driver_process.start()
    app_process.start()
    
    app_process.join()
    test_basic_driver_process.join()