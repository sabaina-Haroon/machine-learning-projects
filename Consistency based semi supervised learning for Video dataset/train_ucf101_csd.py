from data import *
from layers.modules import MultiBoxLoss
# from ssd_consistency import build_ssd_con
from csd import build_ssd_con
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import copy
from dataloader_ucf101 import UCF101DataLoader
import pickle
from matplotlib import pyplot as plt
import pickle


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='UCF101',
                    type=str)
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='model_rgb.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    # loading annotation data
    with open('pyannot.pkl', 'rb') as fp:
        ann_file = pickle.load(fp)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    finish_flag = True

    # loading configuration for our dataset
    cfg = UCF101

    while (finish_flag):

        # load single shot consistency network
        ssd_net = build_ssd_con('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
        #net.load_state_dict(torch.load('weights/ssd300_UCF_.pth'))

        if torch.cuda.device_count() > 1:
            print('using multiple gpus', torch.cuda.device_count())
            net = nn.DataParallel(ssd_net)

        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        #loading imagenet pretrained weights for i3d
        i3d_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.i3d.load_state_dict(i3d_weights)

        if args.cuda:
            net = net.cuda()

        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

        # MultiBoxLoss used for supervised loss calculation
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, args.cuda)

        # loss for both supervised and unsupervised data
        conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        supervised_flag = 1
        loss_ = []
        consistency_loss_ = []
        loss_c_ = []
        loss_l_ = []

        step_index = 0

        if args.visdom:
            vis_title = 'SSD.PyTorch on ' + 'UCF101'
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend, viz)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, viz)

        total_un_iter_num = 0

        _batchsize = args.batch_size

        print('Loading the dataset...')
        video_dataset = UCF101DataLoader(ann_file=ann_file)

        video_data_loader = data.DataLoader(video_dataset, _batchsize,
                                            num_workers=0,  # num_workers=args.num_workers,
                                            shuffle=True,
                                            pin_memory=False, drop_last=True)

        batch_iterator = iter(video_data_loader)

        epoch_size = int(9537 / args.batch_size)

        for iteration in range(args.start_iter, cfg['max_iter']):
            if iteration != 0 and (iteration % epoch_size == 0):
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            try:
                # load images, annotations against a batch ; semis has supervised/unsupervised flags
                videos, targets, semis = next(batch_iterator)


            except StopIteration:
                print('data not loaded')
                print('epoch', epoch)
                supervised_flag = 0
                video_dataset = UCF101DataLoader(ann_file=ann_file)

                video_data_loader = data.DataLoader(video_dataset, _batchsize,
                                                    num_workers=0,  # num_workers=args.num_workers,
                                                    shuffle=True,
                                                    pin_memory=False, drop_last=True)
                batch_iterator = iter(video_data_loader)
                videos, targets, semis = next(batch_iterator)

            if args.cuda:
                videos = Variable(videos.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                videos = Variable(videos)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()

            # forward pass through network returns confidence and localization maps with their flipped versions and
            # out contains confidence , localization and prior predictions for supervised loss calculation
            out, conf, conf_flip, loc, loc_flip = net(videos)

            sup_image_binary_index = np.zeros([len(semis), 1])

            # from ground truth information retain only valid targets , find indices of supervised videos
            for super_image in range(len(semis)):
                if int(semis[super_image]) == 0:
                    sup_image_binary_index[super_image] = 1
                else:
                    sup_image_binary_index[super_image] = 0

                if int(semis[len(semis) - 1 - super_image]) == 1:
                    del targets[len(semis) - 1 - super_image]

            sup_image_index = np.where(sup_image_binary_index == 1)[0]

            loc_data, conf_data, priors = out

            # collect only supervised network outputs of confidence and localization
            if len(sup_image_index) != 0:
                loc_data = loc_data[sup_image_index, :, :]
                conf_data = conf_data[sup_image_index, :, :]
                output = (
                    loc_data,
                    conf_data,
                    priors
                )

            # backprop
            # loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))

            for i in range(0, len(targets)):
                targets[i] = torch.unsqueeze(targets[i], 0)

            if len(sup_image_index) != 0:
                try:
                    loss_l, loss_c = criterion(output, targets)
                    del output
                except:
                    break
                    print('Supervised loss exception')

            # sampling the confidence scores based upon background elimination
            # classes whose confidence is less than background class confidence are sampled out
            sampling = True
            if sampling is True:
                conf_class = conf[:, :, 1:].clone()
                background_score = conf[:, :, 0].clone()
                each_val, each_index = torch.max(conf_class, dim=2)
                mask_val = each_val > background_score
                mask_val = mask_val.data

                mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
                mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

                conf_mask_sample = conf.clone()
                loc_mask_sample = loc.clone()
                conf_sampled = conf_mask_sample[mask_conf_index].view(-1, 25)
                loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

                conf_mask_sample_flip = conf_flip.clone()
                loc_mask_sample_flip = loc_flip.clone()
                conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 25)
                loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

            if mask_val.sum() > 0:
                ## JSD !!!!!1
                conf_sampled_flip = conf_sampled_flip + 1e-7
                conf_sampled = conf_sampled + 1e-7
                consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(),
                                                                     conf_sampled_flip.detach()).sum(-1).mean()
                consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                                                                     conf_sampled.detach()).sum(-1).mean()
                consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

                ## LOC LOSS
                consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
                consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
                consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
                consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

                consistency_loc_loss = torch.div(
                    consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                    4)

                del consistency_conf_loss_a
                del consistency_conf_loss_b
                del consistency_loc_loss_h
                del consistency_loc_loss_w
                del consistency_loc_loss_x
                del consistency_loc_loss_y


            else:
                consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))

            consistency_loss = torch.div(consistency_conf_loss, 2) + consistency_loc_loss


            # giving weightage to unsupervised loss in such a way that model is first trained for
            # supervised loss , and when it comes down near to consistency loss both are now learnt equally

            ramp_weight = rampweight(iteration)
            consistency_loss_actual = consistency_loss.item()
            consistency_loss = torch.mul(consistency_loss, ramp_weight)


            if supervised_flag == 1:
                loss = loss_l + loss_c + consistency_loss
            else:
                if len(sup_image_index) == 0:
                    loss = consistency_loss
                else:
                    loss = loss_l + loss_c + consistency_loss

            if loss.data > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1 = time.time()
            if (len(sup_image_index) == 0):
                loss_l.data = Variable(torch.cuda.FloatTensor([0]))
                loss_c.data = Variable(torch.cuda.FloatTensor([0]))
            else:
                loc_loss += loss_l.data  # [0]
                conf_loss += loss_c.data  # [0]

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f || consistency_loss : %.4f ||' % (
                    loss.data, consistency_loss.data), end=' ')
                print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, lr : %.4f, super_len : %d\n' % (
                    loss.data, loss_c.data, loss_l.data, consistency_loss.data, float(optimizer.param_groups[0]['lr']),
                    len(sup_image_index)))
                    
            if iteration % 100 == 0:
                torch.save(ssd_net.state_dict(), 'weights/ssd300_UCF_' + '.pth')

            if (float(loss) > 1000):
                print('loss greater than 100')
                break

            loss_.append(loss.item())
            consistency_loss_.append(consistency_loss_actual)
            loss_c_.append(loss_c.item())
            loss_l_.append(loss_l.item())

            if args.visdom:
                update_vis_plot(iteration, loss_l.data, loss_c.data,
                                iter_plot, epoch_plot, 'append', viz)

            if iteration != 0 and (iteration + 1) % 40000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_UCF_' +
                           repr(iteration + 1) + '.pth')

            del loss_l
            del loss_c
            del consistency_loss
            del consistency_conf_loss
            del conf_class
            del conf_flip
            del loc
            del loc_flip
            del out
            torch.cuda.empty_cache()

        # print('Saving state, iter:', iteration)
        # torch.save(ssd_net.state_dict(), 'weights/ssd300_UCF_' + '.pth')

        # torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')
        print('-------------------------------\n')
        print(loss.data)
        print('-------------------------------')

        if ((iteration + 1) == cfg['max_iter']):

            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_UCF_' + '.pth')

            iterations_ = list(range(args.start_iter, cfg['max_iter'], 10))
            total_epochs = int(cfg['max_iter'] / epoch_size)
            loss_ep = []
            loss_c_ep = []
            loss_l_ep = []
            consistency_loss_ep = []

            np.save("loss_", loss_)
            np.save("loss_c", loss_c_)
            np.save("loss_l", loss_l_)
            np.save("consistency_loss_", consistency_loss_)

            for ep in range(0, total_epochs):
                sum_loss = 0
                sum_loss_l = 0
                sum_loss_c = 0
                sum_cons = 0
                for j in range(epoch_size * ep, (ep + 1) * epoch_size):
                    sum_loss += loss_[j]
                    sum_loss_l += loss_l_[j]
                    sum_loss_c += loss_c_[j]
                    sum_cons += consistency_loss_[j]

                loss_ep.append(sum_loss/epoch_size)
                loss_c_ep.append(sum_loss_c/epoch_size)
                loss_l_ep.append(sum_loss_l/epoch_size)
                consistency_loss_ep.append(sum_cons/epoch_size)

            epochs = list(range(1, total_epochs + 1))

            # loss
            plt.plot(epochs, loss_ep)
            plt.plot(epochs, loss_c_ep)
            plt.plot(epochs, loss_l_ep)

            plt.title('supervised loss')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend(['overall', 'supervised conf', 'supervised localization'], loc='upper right')
            plt.savefig('loss_plot.jpg')
            plt.show()

            plt.plot(epochs, consistency_loss_ep)
            plt.title('unsupervised loss')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend(['consistency loss'], loc='upper right')
            plt.savefig('unloss_plot.jpg')
            plt.show()
            finish_flag = False



# weight for unsupervised loss increases very slowly , till 7500 iteration
# weightage decreases after 7500 iterations
def rampweight(iteration):
    ramp_up_end = 2400
    ramp_down_start = 7500

    if (iteration < ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2))
    elif (iteration > ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (900 - iteration) / 150), 2))
    else:
        ramp_weight = 1

    if (iteration == 0):
        ramp_weight = 0

    return ramp_weight

# learning rate steps are [100, 8000, 9000]
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend, viz):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type, viz,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':

    train()
