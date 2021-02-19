from __future__ import print_function

import pickle
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from csd import build_ssd_con
from test_set_dataloader import UCF101DetectorData
from layers.box_utils import jaccard, point_form

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_UCF_.pth',
                    type=str, help='Trained state_dict file path to open')
# parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

labelmap = {1: "Basketball",
            2: "BasketballDunk",
            3: "Biking",
            4: "CliffDiving",
            5: "CricketBowling",
            6: "Diving",
            7: "Fencing",
            8: "FloorGymnastics",
            9: "GolfSwing",
            10: "HorseRiding",
            11: "IceDancing",
            12: "LongJump",
            13: "PoleVault",
            14: "RopeClimbing",
            15: "SalsaSpin",
            16: "SkateBoarding",
            17: "Skiing",
            18: "Skijet",
            19: "SoccerJuggling",
            20: "Surfing",
            21: "TennisSwing",
            22: "TrampolineJumping",
            23: "VolleyballSpiking",
            24: "WalkingWithDog"}


def test_net(save_folder, net, cuda, testset, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'test2.txt'
    num_images = len(testset)

    total_precision = 0
    total_recall = 0
    total_images_detected = 0

    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        if testset.if_video_in_annot(i):
            x, annotation, vid_id = testset.__getitem__(i)
            annotation = annotation.astype('int')
            img = x.clone()
            with open(filename, mode='a') as f:
                f.write('\nGROUND TRUTH FOR: ' + vid_id + '\n')
                f.write('label: ' + labelmap[annotation[0] + 1])
                f.write(' || ' + str(annotation[1]))
                f.write(' || ' + str(annotation[2]))
                f.write(' || ' + str(annotation[3]))
                f.write(' || ' + str(annotation[4]) + '\n')
            if cuda:
                x = x.cuda()

            x = torch.unsqueeze(x, 0)
            y = net(x)  # forward pass
            detections = y.data
            detections = detections[0, 1:(len(labelmap)+1), :, :]
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[3], img.shape[2],
                                  img.shape[3], img.shape[2]])

            predicted = (detections[:, :, 1:] * scale)
            truths = annotation[1:]

            # iou = jaccard(truths, point_form(torch.unsqueeze(predicted, 0)))

            pred_num = 0

            class_pred = []
            boxes = []

            for i_ in range(detections.size(0)):
                j = 0
                while detections[i_, j, 0] >= 0.4:
                    if pred_num == 0:
                        with open(filename, mode='a') as f:
                            f.write('PREDICTIONS: ' + '\n')
                    score = detections[i_, j, 0]
                    label_name = labelmap[i_ + 1]

                    pt = (detections[i_, j, 1:] * scale).cpu().numpy()
                    class_pred.append(i_)
                    boxes.append(pt)
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1
                    with open(filename, mode='a') as f:
                        f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                                str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                    j += 1
            tp = 0
            fp = 0
            fn = 0

            if len(class_pred) > 1:
                for bx in range(0, len(class_pred)):
                    ioU = bb_intersection_over_union(truths, boxes[bx])

                    # if IoU ≥0.5, classify the object detection as True Positive(TP)
                    if ioU > 0.5 and class_pred[bx] == annotation[0]:

                        tp += 1

                    # if Iou <0.5, then it is a wrong detection and classify it as False Positive(FP)

                    elif ioU < 0.5 and class_pred[bx] == annotation[0]:
                        fp += 1

                    elif ioU >= 0.5 and class_pred[bx] != annotation[0]:
                        fn += 1

                print('tp', tp)
                print('false', fp)

                Precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)

                total_precision += Precision
                total_recall += recall

                total_images_detected += 1

    mAp = total_precision / total_images_detected

    avg_recall = total_recall / total_images_detected

    print('mAp', mAp)
    print("avg_recall", avg_recall)

    # for i_ in range(detections.size(0)):
    #     j = 0
    #     while detections[i_, j, 0] >= 0.4:
    #         if pred_num == 0:
    #             with open(filename, mode='a') as f:
    #                 f.write('PREDICTIONS: ' + '\n')
    #         score = detections[i_, j, 0]
    #         label_name = labelmap[i_]
    #         pt = (detections[i_, j, 1:] * scale).cpu().numpy()
    #         coords = (pt[0], pt[1], pt[2], pt[3])
    #         pred_num += 1
    #         with open(filename, mode='a') as f:
    #             f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
    #                     str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
    #         j += 1


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def test_voc():
    # load net
    num_classes = len(labelmap) + 1  # +1 background
    net = build_ssd_con('test', 300, 25)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    with open('pyannot.pkl', 'rb') as fp:
        ann_file = pickle.load(fp)
    # load data
    testset = UCF101DetectorData(ann_file=ann_file)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
