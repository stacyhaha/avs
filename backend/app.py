import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process, Queue
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


Q = Queue(maxsize = 50)
Destination = [None]

@app.route("/backend/img", methods=["POST"])
def get_img():
    print("get img from car")
    img = request.json
    print(img)
    Q.put(img)

@app.route("/backend/destination", methods=["POST"])
def get_destination():
    print("get destination")
    destination = request.json
    print(destination)
    # Destination[0] = destination
    return {"status": "ok"}


def run_flask():
    app.run(host="0.0.0.0", port=930)


if __name__ == "__main__":
    app_process = Process(target=run_flask)
    backend_process = Process(target=main, kwargs={"img_queue": Q, "destination": Destination})
    app_process.start()
    backend_process.start()

    app_process.join()
    backend_process.join()


    