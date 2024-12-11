import cv2
import numpy as np
from PIL import Image


def resize_and_center(image, target_width, target_height):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_img = cv2.resize(img, (new_width, new_height),
                             interpolation=cv2.INTER_CUBIC)

    padded_img = np.ones((target_height, target_width, 3),
                         dtype=np.uint8) * 255

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    padded_img[top:top + new_height, left:left + new_width] = resized_img

    return Image.fromarray(padded_img)
