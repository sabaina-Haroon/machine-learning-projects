import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt


def add_hail(image_file, name):
    mask_path = 'hail_mask_4.jpg'

    img = cv2.imread(image_file)
    mask = cv2.imread(mask_path)

    img = np.array(img)
    mask = np.array(mask)

    brightness_coefficient = 0.78  # rainy days are usually shady
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_coefficient
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    img = np.array(img)

    max_ = np.argmax(mask)
    min_ = np.argmin(mask)
    width, height = img.shape[:2]
    #mask = cv2.resize(mask, (height, width), interpolation=cv2.INTER_AREA)

    th = 50
    bin = mask.copy()
    bin = (mask > th) * 255

    ind = np.where(bin == [255, 255, 255])

    w = len(ind[0])

    overlay = img.copy()

    for i in range(0,w):
        x = ind[0][i]
        y = ind[1][i]
        z = ind[2][i]

        overlay[x, y, z] = mask[x, y, z]

    path = "Input/Harmonization/"
    cv2.imwrite(path + name.split('.')[0] + '_hail_naive_mask.png', bin.astype(dtype='uint8'))
    cv2.imwrite(path + name.split('.')[0] + '_hail_naive.png', overlay.astype(dtype='uint8'))

import os

input_name = []

path = 'Input/Images'

names = os.listdir(path)

for n in names:
    add_hail('Input/Images/' + n, n)
    print('generated template for ' + n)