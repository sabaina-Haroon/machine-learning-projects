import numpy as np
from matplotlib import pyplot as plt
import cv2
from Classifier import classifier
from detector import _detector


def add_bbox(img, labels, pts, w=15, h=6, c=(255, 255, 0)):
    for i in range(0, len(pts)):
        x = pts[i][0]
        y = pts[i][1]
        text = labels[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), c, 1)
        cv2.putText(img, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, lineType=cv2.LINE_AA)

    imgplot = plt.imshow(img)
    plt.show()


# load image
def ATR(img, boundingbox, category):
    boundingbox = np.array(boundingbox)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = _detector(gray_img)
    predicted_labels,pts_classified = classifier(pts, img)
    add_bbox(img, predicted_labels, pts_classified)

    # calculating true labels against points
    actual_label = []
    for pt in pts_classified:
        if inside_bbx(pt, boundingbox):
            actual_label.append(category)
        else:
            actual_label.append('Clutter')

    print('Actual  Predicted')
    print('______  _________')
    for i in range(0, len(predicted_labels)):
        print(actual_label[i] + '   ' + predicted_labels[i])



    # to do display image


def inside_bbx(pt, boundingbox):  # bbx[xstart, xend , ystart, yend]
    if boundingbox[0] <= pt[0] <= boundingbox[1] and boundingbox[2] <= pt[1] <= boundingbox[3]:
        return True
    else:
        return False


if __name__ == "__main__":
    pass


img = cv2.imread('cegr01923_0005_701.jpg')
# xstart , xend , ystart, yend
boundingboximage1 = [51, 73, 93, 103]
category_image1 = 'BTR70'

# this will print output or display it
ATR(img, boundingboximage1, category_image1)

img = cv2.imread('cegr02003_0002_1.jpg')  # 254 264 256 264#101
# xstart , xend , ystart, yend
boundingboximage1 = [190, 204, 99, 105]
category_image1 = 'SUV'

# this will print output or display it
ATR(img, boundingboximage1, category_image1)





