import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import main
import sys
from multiprocessing import Process

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

app = Flask(__name__)
CORS(app)

STATUS = "no_mission"  
# STATUS 有两种状态，有任务和没任务，mission 或者 no_mission
# mission指用户确定了目的地
# no_mission指用户没有指定目的地



@app.route("/frontend/car_location", methods=["POST"])
def show_car_location():
    print("get car_location")
    car_location = request.json
    print(car_location)
    #show car location in the interface


@app.route("/frontend/path", methods=["POST"])
def show_path():
    print("get path")
    path = request.json
    print(path)
    #show path in the interface


def run_flask():
    app.run(host="0.0.0.0", port=932)

if __name__ == "__main__":
    app_process = Process(target=run_flask)
    backend_process = Process(target=main)
    app_process.start()
    backend_process.start()

    app_process.join()
    backend_process.join()
