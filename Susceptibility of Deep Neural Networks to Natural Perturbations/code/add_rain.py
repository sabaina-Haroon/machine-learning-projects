import numpy as np
import cv2
from matplotlib import pyplot as plt


def generate_random_lines(imshape, slant, drop_length, density):
    drops = []
    for i in range(density):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))
        else:
            x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))

    return drops


def add_rain(image, drop_length, drop_width, alpha, density, slant):
    imshape = image.shape
    overlay = image.copy()
    # slant_extreme = 10
    # slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_color = (175, 195, 204)  # a shade of gray
    rain_drops = generate_random_lines(imshape, slant, drop_length, density)
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)

        # Following line overlays transparent rectangle over the imagehm
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image_new


def add_rain_affect(IMAGE_FILE, name):

    img = cv2.imread(IMAGE_FILE)

    brightness_coefficient = 0.82  # rainy days are usually shady
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_coefficient
    image_RGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    drop_length = 10
    drop_width = 1
    alpha = 0.68  # Transparency factor.
    density = 4500
    slant = 0
    rain_img = add_rain(image_RGB, drop_length, drop_width, alpha, density, slant)

    drop_length = 10
    drop_width = 1
    alpha = 0.68  # Transparency factor.
    density = 4000
    rain_img = add_rain(rain_img, drop_length, drop_width, alpha, density, slant)


    drop_length = 15
    drop_width = 1
    alpha = 0.68  # Transparency factor.
    density = 4800
    rain_img = add_rain(rain_img, drop_length, drop_width, alpha, density, slant)

    rain_img = cv2.blur(rain_img, (3, 3))
    cv2.imwrite(name.split('.')[0] + 'rain.png', rain_img)
    cv2.imshow('rain', rain_img)
    cv2.waitKey(0)


import os

input_name = []

path = 'Input/Images'

names = os.listdir(path)

for n in names:
    add_rain_affect('Input/Images/' + n, n)
    print('generated template for ' + n)