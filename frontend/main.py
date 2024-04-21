import sys
import logging 
import requests

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

backend_url = "http://localhost:930"
destination_url = f"{backend_url}/backend/destination"

def main():
    # 启动界面
    # 加载地图
    # 尝试实现3个按钮: start, sent_destination, arrive
    res = requests.post(destination_url, json={"destination": (0, 0)})
    logger.info(res.status_code)

    return