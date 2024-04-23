from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np

def transfer_loc_from_centi_to_grid_map(centi_loc):
    # return coordination in grid map
    return (2, 3)

def decompress_base64_to_image(base64_data):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image

def draw_path_on_image(image_path, path , output_path , line_color=(255, 0, 0), line_width=5):
    image = Image.open(image_path)
    
    draw = ImageDraw.Draw(image)
    #这里得按照cell的大小缩放一下
    adjusted_path = [(node.x * 2, node.y * 2) for node in path]
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