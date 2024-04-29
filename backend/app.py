import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process, Queue, Value
from main import main
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

app = Flask(__name__)
CORS(app)

def run_flask(Q, Destination_X, Destination_Y):
    @app.route("/backend/img", methods=["POST"])
    def get_img():
        print("get img from car")
        img = request.json
        print(img)
        Q.put(img)

    @app.route("/backend/destination", methods=["POST"])
    def get_destination():
        print("get destination")
        destination = request.json["destination"]
        Destination_X.value = destination[0]
        Destination_Y.value = destination[1]
        print("destination", Destination_X.value, Destination_Y.value)
        return {"status": "ok"}
    
    app.run(host="0.0.0.0", port=930)


if __name__ == "__main__":
    Q = Queue(maxsize = 50)
    Destination_X = Value("i", -1)
    Destination_Y = Value("i", -1)

    app_process = Process(target=run_flask, args=(Q, Destination_X, Destination_Y))
    backend_process = Process(target=main, args=(Q, Destination_X, Destination_Y))
    app_process.start()
    backend_process.start()

    app_process.join()
    backend_process.join()


    