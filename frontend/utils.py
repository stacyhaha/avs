def transfer_coordi_2_pixel(origin_coor, map_real_size, unit_size: int, map_img_size):
    pixels_per_unit = map_img_size[0] / (map_real_size[0] / unit_size)
    x = int(origin_coor[0] * pixels_per_unit + pixels_per_unit / 2)
    y = int(origin_coor[1] * pixels_per_unit + pixels_per_unit / 2)
    
    return (x, y)
