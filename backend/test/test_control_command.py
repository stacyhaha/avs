import os
import sys
import rospy
import logging
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify
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


def forward():
    chassis.set_velocity(60,90,0)
    rospy.sleep(2)
    chassis.set_velocity(0,0,0)

def backward():
    chassis.set_velocity(60,270,0)
    rospy.sleep(2)
    chassis.set_velocity(0,0,0)   

def turn_left():
    chassis.set_velocity(0,0,30)
    rospy.sleep(5.8)
    chassis.set_velocity(0,0,0)

def turn_right():
    chassis.set_velocity(0,0,-30)
    rospy.sleep(5.8)
    chassis.set_velocity(0,0,0)


def test_command():
    while True:
        import pdb;pdb.set_trace()

if __name__ == "__main__":
    test_command()