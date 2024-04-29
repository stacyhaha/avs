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


def main(img_queue, destination_x, destination_y):
    basic_driver = BasicDriver()
    localizer = Localizer()
    object_detector = ObjectDetector()
    path_planner = PathPlanner()
    path = None

    while True:
        img_queue.empty()
        print("hahah")
        # img_base64 = img_queue.get()
        # img = decompress_base64_to_image(img_base64)

        # todo: 这里之后会改成多进程
        basic_driving_command = basic_driver.drive(1)
        current_location = localizer.localize(1)
        detect_result = object_detector.detect(1)
        current_location_in_grip_map = transfer_loc_from_centi_to_grid_map(current_location)
        try:
            res = requests.post(url_current_position, json={"car_location": current_location_in_grip_map})
        except Exception as e:
            print(e)
        
        if destination_x.value > 0:
            path = path_planner.plan(current_location_in_grip_map, (destination_x.value, destination_y.value))
            res = requests.post(url_path, json={"path": path})
            print('post path')
            print(path)
            logger.info(res.status_code)
        else:
            path = None

        # merge detect, basic_driving, path的指令
        cmd = merge_cmd(basic_driving_command, path, detect_result)
        try: 
            res = requests.post(url_cmd, json={"cmd": cmd})
        except Exception as e:
            print(e)
        time.sleep(0.1)
