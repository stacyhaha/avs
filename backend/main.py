from basic_driver import BasicDriver
from localizer import Localizer
from object_detector import ObjectDetector
from path_planner import PathPlanner
from utils import transfer_loc_from_centi_to_grid_map, decompress_base64_to_image
import requests
import logging
import time
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

frontend_url = "http://localhost:932"
edge_url = "http://localhost:931"

url_current_position = f"{frontend_url}/frontend/car_location"
url_path = f"{frontend_url}/frontend/path"
url_cmd = f"{edge_url}/edge/cmd"

def merge_cmd(basic_driving_command, path, detect_result):
    
    return basic_driving_command


def main(img_queue, destination):
    basic_driver = BasicDriver()
    localizer = Localizer()
    object_detector = ObjectDetector()
    path_planner = PathPlanner()
    path = None

    while not img_queue.empty():
        img_base64 = img_queue.get()
        img = decompress_base64_to_image(img_base64)

        # todo: 这里之后会改成多进程
        basic_driving_command = basic_driver.drive(img)
        current_position = localizer.localize(img)
        detect_result = object_detector.detect(img)
        current_position_in_grip_map = transfer_loc_from_centi_to_grid_map(current_position)
        res = requests.post(url_current_position, json={"position": current_position_in_grip_map})
        logger.info(res.status_code)


        if destination[0] is not None:
            path = path_planner.plan(current_position_in_grip_map, destination[0])
            res = requests.post(url_path, json={"path": path})
            logger.info(res.status_code)
        else:
            path = None

        # merge detect, basic_driving, path的指令
        cmd = merge_cmd(basic_driving_command, path, detect_result)
        res = requests.post(url_cmd, json={"cmd": cmd})
        logger.info(res.status_code)
        time.sleep(0.5)
