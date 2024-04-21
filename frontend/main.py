import sys
import pygame
import os
import logging 
import requests
import io
from PIL import Image
from utils import transfer_coordi_2_pixel
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

backend_url = "http://localhost:930"
destination_url = f"{backend_url}/backend/destination"

window_size = (800, 500)
car_size_height = 40
map_path = "imgs/map.png"
car_path = "imgs/car.png"
map_unit = 5  # 地图划分时，5cm*5cm是最小的栅格单位
map_real_size = (320, 280)  # 地图的尺寸是280mm * 320mm


def main(CAR_LOCATION_X, CAR_LOCATION_Y):
    # 启动界面
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("AVS")

    # 绘制地图
    map_img = Image.open(map_path)
    resized_map = map_img.resize((int(map_img.size[0]/map_img.size[1]*window_size[1]), window_size[1]))
    image_bytes = io.BytesIO()
    resized_map.save(image_bytes, format="png")
    image_bytes.seek(0)
    map_img = pygame.image.load(image_bytes)
    
    # 调整小车尺寸
    car_img = Image.open(car_path)
    resized_car = car_img.resize((int(car_img.size[0]/car_img.size[1]*car_size_height), car_size_height))
    image_bytes = io.BytesIO()
    resized_car.save(image_bytes, format="png")
    image_bytes.seek(0)
    car_img = pygame.image.load(image_bytes)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))  # 白色背景
        screen.blit(map_img, (0, 0))  # 重新绘制地图

        # 实时更新小车的位置
        car_current_location = (CAR_LOCATION_X.value, CAR_LOCATION_Y.value)
        transfered_car_current_location = transfer_coordi_2_pixel(car_current_location, map_real_size, map_unit, map_img.get_rect()[2:])
        screen.blit(car_img, transfered_car_current_location)
        
        pygame.display.update()
        time.sleep(1)


# if __name__ == "__main__":
#     CAR_LOCATION = [[0, 0]]
#     main(CAR_LOCATION)




# def main():
#     
#     # 尝试实现3个按钮: start, sent_destination, arrive
#     res = requests.post(destination_url, json={"destination": (0, 0)})
#     logger.info(res.status_code)

#     return