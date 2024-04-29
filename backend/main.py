import time
import sys
import requests
import logging
from PIL import Image
from basic_driver import BasicDriver
from localizer import Localizer
from object_detector import ObjectDetector
from path_planner import PathPlanner
from multiprocessing import Queue, Value
from utils import transfer_loc_from_centi_to_grid_map, decompress_base64_to_image, compress_image_to_base64


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
    # 融合各指令
    # input example:
    # basic_driving_command: (1, 0, 0) 
    # path: [(current_pos), (next_pos), (next_pos)]
    # detect_result: 

    # priorities: detect_result["emergency"] > basic_driving["turn"] > path["turn"] > basic_driving["straightforward"]
    merged_cmd = [None, None, None]
    if basic_driving_command[1] != 90 or abs(basic_driving_command[-1] - 0) > 1e-4:
        # car is turning
        logger.info("basic driving is turning, cmd equal basic driving")
        merged_cmd = basic_driving_command
    elif len(path)>=5 and abs(path[0][0] - path[5][0]) > 0 and abs(path[0][0] - path[5][0]):
        # path is turning
        logger.info("path is turning, cmd is turning")
        x_change = path[5][0] - path[0][0]
        y_change = path[5][1] - path[0][1]
        if (x_change > 0 and y_change > 0) or (x_change <0 and y_change <0):
            merged_cmd = (0, 0, 20) # car turn at count-clock wise
        elif (x_change > 0 and y_change < 0) or (x_change < 0 and y_change > 0):
            merged_cmd = (0, 0, -20) # car turn at clock wise
    else:
        merged_cmd = basic_driving_command
    return merged_cmd


def main(img_queue, destination_x, destination_y):
    basic_driver = BasicDriver()
    localizer = Localizer()
    object_detector = ObjectDetector()
    path_planner = PathPlanner(map_path="data/grid.txt")
    path = None

    while True:
        if img_queue.empty():
            time.sleep(0.1)
            continue

        img_base64 = img_queue.get()
        img = decompress_base64_to_image(img_base64)

        # todo: 这里之后会改成多进程
        basic_driving_command = basic_driver.drive(img)
        logger.info(basic_driving_command)

        current_position = localizer.localize(img)
        logger.info(current_position)

        detect_result = object_detector.detect(img)
        logger.info(detect_result)
        current_position_in_grip_map = transfer_loc_from_centi_to_grid_map(current_position)
        
        try:
            res = requests.post(url_current_position, json={"car_location": current_position_in_grip_map})
            logger.info(res.status_code)
        except Exception as e:
            logger.info(f"error infomation: {e}")


        if destination_x.value > 0:  # when destination_x < 0, then don't need to do path planning
            path = path_planner.plan(current_position_in_grip_map, (destination_x.value, destination_y.value))
            res = requests.post(url_path, json={"path": path})
            logger.info(res.status_code)
        else:
            path = None

        # merge detect, basic_driving, path的指令
        cmd = merge_cmd(basic_driving_command, path, detect_result)
        
        try:
            res = requests.post(url_cmd, json={"cmd": cmd})
            logger.info(res.status_code)
        except Exception as e:
            print(e)

        time.sleep(0.1)



if __name__ == "__main__":
    img_queue = Queue(maxsize=20)
    destination_X = Value("i", -1)
    destination_Y = Value("i", -1)
    test_img = Image.open("data/basic_drive_test.png")
    base64 = compress_image_to_base64(test_img)
    img_queue.put(base64)
    main(img_queue, destination_X, destination_Y)


