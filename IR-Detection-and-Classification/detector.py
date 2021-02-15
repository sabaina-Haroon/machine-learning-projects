import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


def _detector(img):
    array=[]
    points=[]
    orb = cv2.ORB_create()
    orb.setFastThreshold(50)
    h,w=img.shape
    boxh=int(h/5)
    boxw=int(w/5)
    kp, des = orb.detectAndCompute(img,None)
    x1=0
    y1=0
    while y1<=h:
        while x1<=w:
            for i in range(len(kp)):
                xf,yf=kp[i].pt
                xf=int(xf)
                yf=int(yf)
                if xf>=y1 and xf<=y1+boxh and yf>=x1 and yf<=x1+boxw:
                    array.append((xf,yf))
            if len(array)>=1:
                point=int(len(array)/2)
                a,b=array[int(point)]
                points.append((a,b))
            x1=x1+boxw
            array.clear()
        y1=y1+boxh
        x1=0
    return points
