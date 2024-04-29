import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from col_img import col_img
import sys
from multiprocessing import Process, Queue
import hiwonder

chassis = hiwonder.MecanumChassis()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

app = Flask(__name__)
CORS(app)

@app.route("/edge/cmd", methods=["POST"])
def cmd_car():
    # 收到关于小车的控制，并实现对小车的控制
    print("get cmd")
    cmd = request.json["cmd"]
    print(cmd)
    chassis.set_velocity(cmd)

def run_flask():
    app.run(host="0.0.0.0", port=931)


if __name__ == "__main__":
    app_process = Process(target=run_flask)
    edge_process = Process(target=col_img)

    app_process.start()
    edge_process.start()

    app_process.join()
    edge_process.join()
