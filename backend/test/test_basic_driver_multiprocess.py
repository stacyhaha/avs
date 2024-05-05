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

from basic_driver import BasicDriver
from utils import decompress_base64_to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
fh = logging.FileHandler("test_car.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)  # 把formater绑定到fh上
logger.addHandler(fh)
logger.warning("test")  

app = Flask(__name__)
CORS(app)


def run_flask(shared_data):
    logger = logging.getLogger()
    fh = logging.FileHandler("flask.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)  # 把formater绑定到fh上
    logger.addHandler(fh)
    logger.warning("test")  


    @app.route("/backend/img", methods=["POST"])
    def get_img():
        logger.info("get img")
        img = request.json["img"]
        shared_data.value = img.encode()
        return {"status": "ok"}
    app.run(host="0.0.0.0", port=930)


def test_basic_driver(shared_data):

    logger.info("upload basic driver")
    basic_driver = BasicDriver()
    i = 0
    while True:
        logger.info(f"shared_data length {len(shared_data.value)}")
        if len(shared_data.value) > 0:
            logger.info("basic drive get pic")
            img = shared_data.value.decode()
            img = decompress_base64_to_image(img)
            
            st = time.time()
            cmd = basic_driver.drive(img)
            logger.info(f"driving car process one pic {time.time()-st:.3f} s")
            logger.info(str(cmd))
            try:
                res = requests.post("http://172.17.0.1:9831/edge/cmd", json={"cmd":cmd})
            except:
                pass
            
            try:
                logger.info("saving")
                img = np.array(img)
                opencv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{i}.png", opencv_image)
                
                if  i == 0:
                    with open("cmd.txt", "w") as f:
                        f.write()
                with open("cmd.txt", "a") as f:
                    f.write(f"{i}")
                i += 1


                
                cv2.imshow("Image", opencv_image)
                cv2.waitKey(2000)
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




