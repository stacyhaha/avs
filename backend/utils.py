import sys
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
import logging 

import sys

# 将 stdout 和 stderr 重定向到终端
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def transfer_loc_from_centi_to_grid_map(centi_loc):
    
    # return coordination in grid map
    return (2, 3)

def decompress_base64_to_image(base64_data):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image


def compress_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    base64_encoded = base64.b64encode(buffer).decode("utf-8")
    return base64_encoded

def draw_path_on_image(image_path, path , output_path , line_color=(255, 0, 0), line_width=5):
    image = Image.open(image_path)
    
    draw = ImageDraw.Draw(image)
    #这里得按照cell的大小缩放一下
    adjusted_path = [(x * 2, (image.height - (y * 2)) - 1) for x, y in path]
    #adjusted_path = [(x * 2, y * 2) for x, y in path]
    draw.line(adjusted_path, fill=line_color, width=line_width)

    image.save(output_path)
    image.show()

def identify_obstacles(image_path, grid_size):

    #readme:
    #image_path = 'Map_v1.jpg'
    #grid_size = (255, 292)
    #obstacle_grid = identify_obstacles(image_path, grid_size)

    image = Image.open(image_path)
    gray_image = image.convert('L')
    gray_image.save('gray_image.jpg')  # 保存灰度图像

    grid_height, grid_width = grid_size
    print(grid_height, grid_width)
    grid = np.zeros(grid_size, dtype=int)
    
    img_width, img_height = gray_image.size
    cell_width = img_width // grid_width
    cell_height = img_height // grid_height

    for i in range(grid_height):
        for j in range(grid_width):
            left = j * cell_width
            upper = i * cell_height
            right = min((j + 1) * cell_width, img_width)
            lower = min((i + 1) * cell_height, img_height)

            cell = gray_image.crop((left, upper, right, lower))
            non_road_pixels = sum(pixel < 200 for pixel in cell.getdata())

            if non_road_pixels > 0.5 * (right-left) * (lower-upper):
                grid[i, j] = 1

    np.savetxt('grid.txt', grid, fmt="%d")  # 保存网格数据为文本文件
    return grid

def check_obstacle(map_array, x, y, high):
    # 调整 y 坐标来匹配 AStar 中使用的坐标系统
    actual_y = high - y - 1
    return map_array[actual_y, x] == 1




def draw_map(grid, path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, cmap='Greys', origin='lower')  # 使左下角为原点

    if path:
        # 从 Node 对象中提取 x 和 y 坐标
        xs, ys = zip(*[(node.x, node.y) for node in path])
        ax.plot(xs, ys, marker='o', color='red')  # 绘制路径

    plt.show()
