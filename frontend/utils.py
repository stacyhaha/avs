import pygame

def transfer_coordi_2_pixel(origin_coor, map_real_size, unit_size: int, map_img_size):
    pixels_per_unit = map_img_size[0] / (map_real_size[0] / unit_size)
    x = int(origin_coor[0] * pixels_per_unit + pixels_per_unit / 2)
    y = int(origin_coor[1] * pixels_per_unit + pixels_per_unit / 2)
    return (x, y)


def transfer_pixel_2_corrdi(pixel_corri, map_real_size, unit_size, map_img_size):
    width_unit_num = map_real_size[0] / unit_size
    height_unit_num = map_real_size[1] / unit_size
    width_unit = int( pixel_corri[0] / map_img_size[0] * width_unit_num)
    height_unit = int(pixel_corri[1] / map_img_size[1] * height_unit_num)
    return (width_unit, height_unit)


def draw_path(screen, path, map_real_size, unit_size, map_img_size):
    unit_pixel = unit_size / map_real_size[0] * map_img_size[0]

    for xy_coori in path:
        
        x_pixel, y_pixel = transfer_coordi_2_pixel(
            xy_coori,
            map_real_size,
            unit_size,
            map_img_size
        )

        pygame.draw.rect(screen, (0, 255, 0), (x_pixel, y_pixel, unit_pixel, unit_pixel))
    return

