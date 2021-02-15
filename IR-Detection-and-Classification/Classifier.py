import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
from random import shuffle
import cv2

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def get_categoryname(num):
    lookup = {'1': 'MAN', '2': 'PICKUP', '3': 'SUV', '4': 'BTR70', '5': 'BRDM2', '6': 'BMP2', '7': 'T72',
              '8': 'ZSU23', '9': '2S3', '10': 'D20', '11': 'Clutter'}
    return lookup[str(num)]


def classifier(pts, img):
    rx = 20
    ry = 8

    # Model reconstruction
    model = load_model('atr_model.h5')

    pts = np.array(pts)
    size = pts.shape[0]
    w, h = img.shape[:2]

    # cropping sub images around points
    test = []
    pts_new = []
    for i in range(0, size):
        cx = pts[i, 0]
        cy = pts[i, 1]

        startx = (np.abs(cx - rx)).astype('int32')
        endx = (cx + rx)
        starty = (np.abs(cy - ry)).astype('int32')
        endy = (cy + ry)

        newimg = img[starty:endy, startx:endx, :]
        dim = (7 * rx, 7 * ry)
        # resize image
        resized = cv2.resize(newimg, dim, interpolation=cv2.INTER_AREA)

        test.append(resized)
        pts_new.append((cx, cy))

    test = np.array(test)
    test = test.astype('float32')

    # bringing subimages to zero centre
    for i in range(0, test.shape[0]):
        sum_ = test[i].sum()
        totalpixels = test.shape[1] * test.shape[2] * 3
        mean = sum_ / totalpixels
        test[i] -= mean
    test /= 255

    y_pred = model.predict(test, verbose=2)
    pred = np.argmax(y_pred, axis=1)
    pred = pred + 2
    label_predicted= []

    size = pred.shape[0]
    for i in range(0, size):
        label_predicted.append(get_categoryname(pred[i]))

    return label_predicted, pts_new
