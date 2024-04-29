from PIL import Image
from io import BytesIO
import base64

def transfer_loc_from_centi_to_grid_map(centi_loc):
    # return coordination in grid map
    return (2, 3)

def decompress_base64_to_image(base64_data):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image
