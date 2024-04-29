import sys
import pygame
import os
import logging 
import requests
import io
from PIL import Image
from utils import transfer_coordi_2_pixel, transfer_pixel_2_corrdi, draw_path
import time
from multiprocessing import Value

clock = pygame.time.Clock()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__file__)

backend_url = "http://localhost:930"
destination_url = f"{backend_url}/backend/destination"

FPS = 60
window_size = (800, 500)
car_size_height = 40
map_path = "imgs/map.png"
car_path = "imgs/car.png"
map_unit = 5  # 地图划分时，5cm*5cm是最小的栅格单位
map_real_size = (320, 280)  # 地图的尺寸是280mm * 320mm

status = "preparing"  
destination_pos = [0, 0]
# preparing: 点击开始按钮之前
# after_clicking_start: 点击开始按钮之后，但未确定终止位置
# processing: 有起始，终止位置，更新路径 

def main(CAR_LOCATION_X, CAR_LOCATION_Y, PATH):
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
    
    # 绘制小车
    car_img = Image.open(car_path)
    resized_car = car_img.resize((int(car_img.size[0]/car_img.size[1]*car_size_height), car_size_height))
    image_bytes = io.BytesIO()
    resized_car.save(image_bytes, format="png")
    image_bytes.seek(0)
    car_img = pygame.image.load(image_bytes)

    # 绘制开始按钮
    start_button = pygame.Rect(600, 50, 170, 50)
    start_button_color = (0, 255, 0)
    font = pygame.font.SysFont(None, 40)
    start_button_text = font.render("start", True, (0, 0, 0))

    # 绘制取消按钮
    cancel_button = pygame.Rect(600, 300, 170, 50)
    cancel_button_color = (0, 255, 0)
    cancel_button_text = font.render("cancel", True, (0, 0, 0))

    buffer_surface = pygame.Surface(window_size)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        buffer_surface.fill((255, 255, 255))  # 白色背景
        buffer_surface.blit(map_img, (0, 0))  # 重新绘制地图
        
        # 实时更新小车的位置
        car_current_location = (CAR_LOCATION_X.value, CAR_LOCATION_Y.value)
        transfered_car_current_location = transfer_coordi_2_pixel(car_current_location, map_real_size, map_unit, map_img.get_rect()[2:])
        buffer_surface.blit(car_img, transfered_car_current_location)
        
        global status
        if status == "preparing":
            start_button_color = (0, 255, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    pygame.mouse.set_cursor(*pygame.cursors.diamond)
                    status = "after_clicking_start"

        elif status == "after_clicking_start":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if map_img.get_rect().collidepoint(event.pos):
                    pygame.mouse.set_cursor(*pygame.cursors.diamond)
                    status = "after_clicking_start"
                    click_pos = event.pos
                    start_button_color = (150, 150, 150)
                    global destination_pos
                    destination_pos = click_pos
                    pygame.draw.circle(screen, (255, 0, 0), click_pos, 15)
                    destination_corri = transfer_pixel_2_corrdi(destination_pos, map_real_size, map_unit, map_img.get_rect()[2:])
                    try:
                        res = requests.post(destination_url, json={"destination": destination_corri})
                    except Exception as e:
                        logger.info(e)
                    logger.info("post destination")
                    pygame.mouse.set_cursor(*pygame.cursors.arrow)
                    status = "processing"

        elif status == "processing":
            pygame.mouse.set_cursor(*pygame.cursors.arrow)
            pygame.draw.rect(buffer_surface, cancel_button_color, cancel_button)  # 绘制开始按钮
            buffer_surface.blit(cancel_button_text, cancel_button_text.get_rect(center = cancel_button.center))  # 绘制开始按钮上的文字
            draw_path(buffer_surface, PATH, map_real_size, map_unit, map_img.get_rect()[2:])
            pygame.draw.circle(buffer_surface, (255, 0, 0), click_pos, 15)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if cancel_button.collidepoint(event.pos):
                    status = "preparing"
                    
                try:
                    res = requests.post(destination_url, json={"destination": [-1, -1]})
                except Exception as e:
                    logger.info(e)
                logger.info("post destination")
            

        pygame.draw.rect(buffer_surface, start_button_color, start_button)  # 绘制开始按钮
        buffer_surface.blit(start_button_text, start_button_text.get_rect(center = start_button.center))  # 绘制开始按钮上的文字
        screen.blit(buffer_surface, (0, 0))
        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":

    main(Value("i", 0), Value("i", 0))




# def main():
#     
#     # 尝试实现3个按钮: start, sent_destination, arrive


#     return