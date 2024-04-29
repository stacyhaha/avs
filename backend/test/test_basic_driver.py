import cv2
import os
import sys
import logging
from PIL import Image
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify
import ctypes
import matplotlib.pyplot as plt
import asyncio

shared_data = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils import decompress_base64_to_image


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

app = Flask(__name__)
CORS(app)


async def run_flask():
    @app.route("/img_test", methods=["POST"])
    def get_img():
        print("get img")
        img = request.json["img"]
        global shared_data
        shared_data = img.encode()
        return {"status": "ok"}

    app.run(host="0.0.0.0", port=930)

async def test_basic_driver():
    while True:
        global shared_data
        if shared_data is not None:
            img = shared_data.decode()
            img = decompress_base64_to_image(img)

            img = np.array(img)
            print("open img")

            
            opencv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 使用 matplotlib 实时显示图像
            plt.imshow(opencv_image)
            plt.draw()
            plt.pause(0.001)  # 短暂暂停，允许图像显示
            plt.clf()  # 清空当前图像，准备显示下一帧

        # await asyncio.sleep(0.1)  # 适当的休眠时间，控制刷新频率


async def main():
    run_flask_task = asyncio.create_task(run_flask())
    test_basic_driver_task = asyncio.create_task(test_basic_driver())
    await run_flask_task 
    await test_basic_driver_task
    


if __name__ == "__main__":
    asyncio.run(main())



