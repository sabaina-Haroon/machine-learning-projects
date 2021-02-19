import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

err_not_np_img= "not a numpy array or list of numpy array"
err_img_arr_empty="Image array is empty"
err_row_zero="No. of rows can't be <=0"
err_column_zero="No. of columns can't be <=0"
err_invalid_size="Not a valid size tuple (x,y)"
err_caption_array_count="Caption array length doesn't matches the image array length"


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_tuple(x):
    return type(x) is tuple


def is_list(x):
    return type(x) is list


def is_numeric(x):
    return type(x) is int


def is_numeric_list_or_tuple(x):
    for i in x:
        if not is_numeric(i):
            return False
    return True


def verify_image(image):
    if is_numpy_array(image):
        pass
    elif is_list(image):
        image_list=image
        for img in image_list:
            if not is_numpy_array(img):
                raise Exception(err_not_np_img)
    else:
        raise Exception(err_not_np_img)


err_snow_coeff = "Snow coeff can only be between 0 and 1"


def snow_process(image, snow_coeff):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 2.5
    imshape = image.shape
    snow_point = snow_coeff  ## increase this for more snow
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = image_HLS[:, :, 1][image_HLS[:, :,
                                                                             1] < snow_point] * brightness_coefficient  ## scale pixel values up for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def add_snow(image, snow_coeff=-1):
    verify_image(image)
    if snow_coeff != -1:
        if snow_coeff < 0.0 or snow_coeff > 1.0:
            raise Exception(err_snow_coeff)
    else:
        snow_coeff = random.uniform(0, 1)
        print(snow_coeff)
    snow_coeff *= 255 / 2
    snow_coeff += 255 / 3
    if is_list(image):
        image_RGB = []
        image_list = image
        for img in image_list:
            output = snow_process(img, snow_coeff)
            image_RGB.append(output)
    else:
        output = snow_process(image, snow_coeff)
        image_RGB = output

    return image_RGB


def add_snow_affect(IMAGE_FILE, name):
    #IMAGE_FILE = 'berlin_000000_000019_leftImg8bit.png'
    #IMAGE_FILE = 'frankfurt_000000_000294_leftImg8bit.png'
    img = cv2.imread(IMAGE_FILE)
    imgg = img.copy()
    # snow = add_snow(img, 0.1316866519353459)
    snow = add_snow(img, 0.20316866519353459)

    th = 200
    bin = snow.copy()
    bin = (snow > th) * 255

    newbin = np.logical_and(bin[:, :, 0] >= 255, bin[:, :, 1] >= 255, bin[:, :, 2] >= 255)

    bin[:, :, 0] = newbin * 255
    bin[:, :, 1] = newbin * 255
    bin[:, :, 2] = newbin * 255
    path  = "Input/Harmonization/"
    cv2.imwrite(path + name.split('.')[0] + '_snow_naive_mask.png', bin.astype(dtype='uint8'))
    #plt.imshow(bin)
    #plt.show()
    overlay = np.where(bin == [255, 255, 255],[255, 255, 255], imgg)

    overlay = overlay.astype(dtype='uint8')
    alpha = 0.06
    added_image = cv2.addWeighted(img, 1 - 10*alpha, overlay, 1 - 5*0.05, 0)
    cv2.imwrite(path + name.split('.')[0] + '_snow_naive.png', added_image)
    #cv2.imshow('snow', added_image)
    #cv2.imshow('real', img)
    #cv2.waitKey(0)

import os

input_name = []

path = 'Input/Images'

names = os.listdir(path)

for n in names:
    add_snow_affect('Input/Images/' + n, n)
    print('generated template for ' + n)