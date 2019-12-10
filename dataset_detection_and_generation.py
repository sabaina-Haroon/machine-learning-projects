import pickle
# import sys
# sys.path.append('/home/bruce/projects/bruce')
# import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import cv2
import random
from random import shuffle


class AtrDataset:
    def __init__(self, sample_dir):
        self.path = sample_dir + '/' + sample_dir + '.p'
        self.imagedir = sample_dir + '/' + 'images'
        self.samples = pickle.load(open(self.path, 'rb'))
        self.info()

    def get_num_targets(self):
        num_targets = 0
        for frame in self.samples:
            num_targets += len(frame['targets'])
        return num_targets

    def _get_categories(self):
        categories = []
        for frame in self.samples:
            for target in frame['targets']:
                if target['category'] not in categories:
                    categories.append(target['category'])
        return categories

    def _get_ranges(self):
        rangecount = {}
        for frame in self.samples:
            if frame['range'] not in rangecount.keys():
                rangecount[frame['range']] = 1
            else:
                rangecount[frame['range']] += 1

        print('Range   Count')
        for range in rangecount.keys():
            print('{:<7}'.format(range), rangecount[range])

    def _get_day(self):
        day = 0
        night = 0
        for frame in self.samples:
            if frame['day']:
                day += 1
            else:
                night += 1
        print('day       night')
        print(day, '  ', night)

    def get_num_frames(self):
        return (self.number_of_frames)

    def info(self):
        self.number_of_frames = len(self.samples)
        self.number_of_targets = self.get_num_targets()
        self.categories = self._get_categories()
        print('frames ', self.number_of_frames)
        print('targets ', self.number_of_targets)
        print('number of classes ', len(self.categories))
        print('categories', self.categories)
        self.count_categories()
        self._get_ranges()
        self._get_day()

    def count_categories(self):
        print('Class    Count')
        for category in self.categories:
            count = 0
            for frame in self.samples:
                for target in frame['targets']:
                    if target['category'] == category:
                        count += 1
            print('{:<8}'.format(category), count)

    def _category_lookup(self, name):
        lookup = {'MAN': 1, 'PICKUP': 2, 'SUV': 3, 'BTR70': 4, 'BRDM2': 5, 'BMP2': 6, 'T72': 7, 'ZSU23': 8, '2S3': 9,
                  'D20': 10, 'MTLB': 11}
        return lookup[name]

    def get_categoryname(self, num):

        lookup = {'1': 'MAN', '2': 'PICKUP', '3': 'SUV', '4': 'BTR70', '5': 'BRDM2', '6': 'BMP2', '7': 'T72',
                  '8': 'ZSU23', '9': '2S3', '10': 'D20', '10': 'MTLB'}
        return lookup[str(num)]

    def showSample(self, idx):
        # shows sample indexed from master file and avi frame with bbox gt overlay

        sample = self.samples[idx]
        print(sample)
        imgfile = self.imagedir + '/' + sample['name'] + '_' + sample['frame'] + '.jpg'
        image = Image.open(imgfile)

        f, ax = plt.subplots(figsize=(10, 10))
        for target in sample['targets']:
            cx = target['center'][0]
            cy = target['center'][1]

            ulx = target['ul'][0]
            uly = target['ul'][1]

            w = (cx - ulx) * 2
            h = (cy - uly) * 2

            rect = patches.Rectangle((ulx, uly), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.imshow(image)
        ax.set_title(sample['name'], fontsize=30)
        plt.show()


def averageradius(dataset):
    radiusX = 0
    radiusY = 0
    for frame in dataset.samples:
        for bbx in frame['targets']:
            cx = bbx['center'][0]
            cy = bbx['center'][1]

            ulx = bbx['ul'][0]
            uly = bbx['ul'][1]

            rx = (cx - ulx)
            ry = (cy - uly)

            radiusX += rx
            radiusY += ry

    radiusX /= dataset.number_of_frames
    radiusY /= dataset.number_of_frames

    return radiusX, radiusY


def _category_lookup(name):
    lookup = {'MAN': 1, 'PICKUP': 2, 'SUV': 3, 'BTR70': 4, 'BRDM2': 5, 'BMP2': 6, 'T72': 7, 'ZSU23': 8, '2S3': 9,
              'D20': 10, 'MTLB': 11}
    return lookup[name]


def _detector(x1, x2, y1, y2, targetCentreX, targetCentreY, img, threshold):

    orb = cv2.ORB_create()
    orb.setFastThreshold(threshold)
    clutter = []
    blobPoints, des = orb.detectAndCompute(img, None)
    target = (0, 0)
    targets = []
    minDistance = 1000000000

    for i in range(len(blobPoints)):
        xblob, yblob = blobPoints[i].pt
        xblob = int(xblob)
        yblob = int(yblob)
        if xblob >= x1 - 1 and xblob <= x2 + 1 and yblob >= y1 - 1 and yblob < y2 + 1:
            distance = (targetCentreX - xblob) * (targetCentreX - xblob) + (targetCentreY - yblob)*(
                targetCentreY - yblob)
            if distance <= minDistance:
                minDistance = distance
                target = (xblob, yblob)

        else:
            clutter.append((xblob, yblob))

    return target, clutter

def _sample(clutter):
    _shuffle = []
    size = len(clutter)
    index_shuf = list(range(size))
    shuffle(index_shuf)
    for i in index_shuf:
        _shuffle.append(clutter[i])

    new_ind = int(np.ceil(size*0.02))


    return _shuffle[0:new_ind]



def createDatasetdetector(dataset, rx, ry, day_or_night):
    x_train = []
    y_train = []
    notdetected = 0
    for sample in dataset.samples:
        if sample['day'] == day_or_night:
            imgfile = dataset.imagedir + '/' + sample['name'] + '_' + sample['frame'] + '.jpg'
            img = cv2.imread(imgfile)
            img = np.array(img)
            w, h = img.shape[:2]

            target = sample['targets']
            target = target[0]

            cx = target['center'][0]
            cy = target['center'][1]

            x1 = cx - rx
            if x1 < 0:
                x1 = 0

            x2 = cx + rx
            if x2 >= h:
                x2 = h - 1

            y1 = cy - ry
            if y1 < 0:
                y1 = 0

            y2 = cy + ry
            if y2 >= w:
                y2 = w - 1

            threshold = 10

            target_pts , clutter = _detector(x1, x2, y1, y2, cx, cy, img, threshold)

            if target_pts != (0,0):
                cx = target_pts[0]
                cy = target_pts[1]

                if (cx - rx) < 0 or (cy - ry) < 0:
                    cx = rx
                    cy = ry
                if (cx + rx) > h or (cy + ry) > w:
                    cx = w - cx - 1
                    cy = h - cy - 1
                if (cx - rx) < 0 or (cy - ry) < 0 or (cx + rx) > h or (cy + ry) > w:
                    skipping_outofbound += 1
                    continue

                startx = (np.abs(cx - rx)).astype('int32')
                endx = (cx + rx)

                starty = (np.abs(cy - ry)).astype('int32')
                endy = (cy + ry)

                newimg = img[starty:endy, startx:endx, :]
                row, col = newimg.shape[:2]

                dim = (7 * rx, 7 * ry)
                # resize image
                resized = cv2.resize(newimg, dim, interpolation=cv2.INTER_AREA)


                x_train.append(resized)

                y = _category_lookup(target['category'])
                y_train.append(y)
            else:
                notdetected += 1

            clutter = _sample(clutter)

            for pt in clutter:

                cx = pt[0]
                cy = pt[1]

                if (cx - rx) < 0 or (cy - ry) < 0:
                    cx = rx
                    cy = ry
                if (cx + rx) > h or (cy + ry) > w:
                    cx = w - cx - 1
                    cy = h - cy - 1
                if (cx - rx) < 0 or (cy - ry) < 0 or (cx + rx) > h or (cy + ry) > w:
                    skipping_outofbound += 1
                    continue

                startx = (np.abs(cx - rx)).astype('int32')
                endx = (cx + rx)

                starty = (np.abs(cy - ry)).astype('int32')
                endy = (cy + ry)

                newimg = img[starty:endy, startx:endx, :]
                row, col = newimg.shape[:2]

                dim = (7*rx, 7*ry)
                #resize image
                resized = cv2.resize(newimg, dim, interpolation=cv2.INTER_AREA)

                x_train.append(resized)
                y_train.append(11)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train


if __name__ == "__main__":
    pass

    dataset_train = AtrDataset('./scaled2500_1to2')
    rx_0, ry_0 = averageradius(dataset_train)
    dataset_test = AtrDataset('./scaled2500_25to35day')
    rx, ry = averageradius(dataset_test)
    #
    rx = (rx + rx_0) / 2
    ry = (ry + ry_0) / 2
    rx = np.ceil(1.5*rx)
    ry = np.ceil(1.5*ry)
    rx = rx.astype('int32')
    ry = ry.astype('int32')

    r = np.ceil(2 * max(rx, ry))

    #r = 32

    x_train_day_2_5, y_train_day_2_5 = createDatasetdetector(dataset_train, rx, ry, 1)

    np.save('x_train_day_2_5.npy', x_train_day_2_5)
    np.save('y_train_day_2_5.npy', y_train_day_2_5)

    x_train_night_2_5, y_train_night_2_5 = createDatasetdetector(dataset_train, rx, ry, 0)

    np.save('x_train_night_2_5.npy', x_train_night_2_5)
    np.save('y_train_night_2_5.npy', y_train_night_2_5)

    x_test_day_3_5, y_test_day_3_5 = createDatasetdetector(dataset_test, rx, ry, 1)

    np.save('y_test_day_3_5.npy', y_test_day_3_5)
    np.save('x_test_day_3_5.npy', x_test_day_3_5)

    x_test_night_3_5, y_test_night_3_5 = createDatasetdetector(dataset_test, rx, ry, 0)

    np.save('y_test_night_3_5.npy', y_test_night_3_5)
    np.save('x_test_night_3_5.npy', x_test_night_3_5)