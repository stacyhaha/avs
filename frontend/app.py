import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import main
import sys
from multiprocessing import Process, Value

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


def run_flask(CAR_LOCATION_X, CAR_LOCATION_Y):
    @app.route("/frontend/car_location", methods=["POST"])
    def show_car_location():
        logger.info("get car_location")
        car_location = request.json["car_location"]
        CAR_LOCATION_X.value = car_location[0]
        CAR_LOCATION_Y.value = car_location[1]
        print("car_lo", CAR_LOCATION_X.value)
        print("car_loc", CAR_LOCATION_Y.value)
        logger.info(f"get car location {car_location}")
        print("id_in app", id(CAR_LOCATION_X))
        return {"status": "ok"}

    
    @app.route("/frontend/path", methods=["POST"])
    def show_path():
        print("get path")
        path = request.json
        print(path)
        #show path in the interface


    app.run(host="0.0.0.0", port=932,)

if __name__ == "__main__":
    CAR_LOCATION_X = Value('i', 0)
    CAR_LOCATION_Y = Value('i', 0)
    app_process = Process(target=run_flask, args=(CAR_LOCATION_X, CAR_LOCATION_Y))
    backend_process = Process(target=main, args=(CAR_LOCATION_X, CAR_LOCATION_Y))
    
    app_process.start()
    backend_process.start()

    app_process.join()
    backend_process.join()
