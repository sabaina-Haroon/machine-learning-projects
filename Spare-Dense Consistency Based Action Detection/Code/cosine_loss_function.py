import numpy as np
import torch
import random
import cv2
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import math
import pickle

def ramp_weight(total_iterations):
    ramp_sup = []
    ramp_self = []
    ramp_semi = []

    ramp_up_end_slf = int(total_iterations/3)
    ramp_up_end_sem = ramp_up_end_slf*2
    for iteration in range(0, total_iterations):

        if (iteration > ramp_up_end_slf) and (iteration < ramp_up_end_sem):
            ramp_weight = 1

        elif iteration > ramp_up_end_slf:
            ramp_weight = math.exp(-30 * math.pow((1 - ramp_up_end_sem / (iteration)), 2))

        else:
            ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end_slf), 2))

        ramp_sup.append(ramp_weight)

        if iteration < ramp_up_end_slf:
            ramp_weight = 1

        else:
            ramp_weight = math.exp(-20 * math.pow((1 - ramp_up_end_slf / (iteration)), 2))

        ramp_self.append(ramp_weight)

        if iteration > ramp_up_end_sem:
            ramp_weight = 1
        else:
            ramp_weight = math.exp(-15 * math.pow((1 - iteration / ramp_up_end_sem), 2))

        ramp_semi.append(ramp_weight)
    '''
    fig, axs = plt.subplots(3,1)
    plt.title('ramp weights for losses')
    axs[0].plot(ramp_self), axs[0].set_title('self-supervised')
    axs[1].plot(ramp_sup),  axs[1].set_title('supervised')
    axs[2].plot(ramp_semi), axs[2].set_title('semi-supervised')
    #plt.legend(['supervised', 'self-supervised', 'semi-supervised'], loc='best')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()'''

    return ramp_sup, ramp_self, ramp_semi

def new_ramp_weight():
    ramp_sup = []
    ramp_self = []
    ramp_semi = []

    ramp_up_end_slf = int(total_iterations/3)
    ramp_up_end_sem = ramp_up_end_slf*2
    for iteration in range(0, total_iterations):

        if (iteration > ramp_up_end_slf) and (iteration < ramp_up_end_sem):
            ramp_weight = 1

        elif iteration > ramp_up_end_slf:
            ramp_weight = math.exp(-30 * math.pow((1 - ramp_up_end_sem / (iteration)), 2))

        else:
            ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end_slf), 2))

        ramp_sup.append(ramp_weight)

        if iteration < ramp_up_end_slf:
            ramp_weight = 1

        else:
            ramp_weight = math.exp(-20 * math.pow((1 - ramp_up_end_slf / (iteration)), 2))

        ramp_self.append(ramp_weight)

        if iteration > ramp_up_end_sem:
            ramp_weight = 1
        else:
            ramp_weight = 0.0

        ramp_semi.append(ramp_weight)
    '''
    fig, axs = plt.subplots(3,1)
    plt.title('ramp weights for losses')
    axs[0].plot(ramp_self), axs[0].set_title('self-supervised')
    axs[1].plot(ramp_sup),  axs[1].set_title('supervised')
    axs[2].plot(ramp_semi), axs[2].set_title('semi-supervised')
    #plt.legend(['supervised', 'self-supervised', 'semi-supervised'], loc='best')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()'''

    temp = ramp_semi
    ramp_semi = ramp_sup
    ramp_sup = temp


    return ramp_sup, ramp_self, ramp_semi

def ramp_weight_epochwise(total_epochs):

    ramp_sup = []
    ramp_self = []
    ramp_semi = []

    ramp_up_end_slf = int(total_epochs/6)
    ramp_up_end_sup = int(total_epochs/2)
    ramp_up_end_sem = int(total_epochs/3)
    for iteration in range(0,total_epochs):



        if (iteration > ramp_up_end_slf) and (iteration < ramp_up_end_sem):
            ramp_weight = 1

        elif iteration >= ramp_up_end_sem:
            ramp_weight = math.exp(-90 * math.pow((1 - ramp_up_end_sem / (iteration)), 2))

        else:
            ramp_weight = math.exp(-25 * math.pow((1 - iteration / ramp_up_end_slf), 2))
        
        ramp_semi.append(ramp_weight)

        if iteration < (ramp_up_end_slf - ramp_up_end_slf*0.30):
            ramp_weight = 1

        else:
            ramp_weight = math.exp(-20 * math.pow((1 - (ramp_up_end_slf - ramp_up_end_slf*0.30) / (iteration+1)), 2))

        ramp_self.append(ramp_weight)


        if iteration > ramp_up_end_sem:
            ramp_weight = 1
        else:
            ramp_weight = 0 #math.exp(-95 * math.pow((1 - iteration / ramp_up_end_sem), 2))

        ramp_sup.append(ramp_weight)
    
    return ramp_sup, ramp_self, ramp_semi

def find_weight(x, y, c):
    w = (float)(y - c) / (float)(y - x)
    return w


def find_interv_weight(fr_intv, trim_vid_cntr):
    interval_count = len(fr_intv)
    for i in range(0, interval_count - 1):
        if fr_intv[i] <= trim_vid_cntr & trim_vid_cntr <= fr_intv[i + 1]:
            weight = find_weight(fr_intv[i], fr_intv[i+1], trim_vid_cntr)
            return weight, i

    if trim_vid_cntr < fr_intv[0]:
        return 1.0, int(0)

    if trim_vid_cntr > fr_intv[len(fr_intv)-1]:
        return 0.0, int(len(fr_intv)-2)



## !!!!!!!! CALCULATE SIMILARITY BETWEEN TWO REPRESENTATIONS !!!!!!!!! ##s
def calc_sim(ref_frame, trimmed_frame):
    # rearrange representations with H x W x Channels
    # ref_f = ref_frame.permute(1, 2, 0)
    # trimmed_f = trimmed_frame.permute(1, 2, 0)

    ref_f = ref_frame
    trimmed_f = trimmed_frame
    # print(f'trimmed_f {trimmed_f.shape}')
    # print(f'ref_f {ref_f.shape}')

    a = torch.squeeze(ref_f.contiguous().view(1, -1))
    b = torch.squeeze(trimmed_f.contiguous().view(1, -1))
    dot_ = torch.dot(a.to(torch.double), b.to(torch.double)).item()
    a_abs = torch.norm(ref_f.type(dtype=torch.double)).item()
    b_abs = torch.norm(trimmed_f.type(dtype=torch.double)).item()
    # if a_abs ==0 or b_abs ==0:
        # print('trim', b_abs )
        # print('full', a_abs )
    similarity = dot_ / (a_abs * b_abs)
    # else:
    # similarity = 1

    return similarity


def calc_loss(ref_frame, trimmed_frame):
    # rearrange representations with H x W x Channels
    # ref_f = ref_frame.permute(1, 2, 0)
    # trimmed_f = trimmed_frame.permute(1, 2, 0)

    criterion_m = nn.MSELoss()
    loss = torch.sqrt(criterion_m(ref_frame, trimmed_frame))

    return loss.item()



# calculate predicted trend :
def calc_sim_trend(full_rep, trim_repr, weight, ref_fr_i, isImage):
    total_fr = full_rep.shape[0]
    sim_arr = []
    x = int(trim_repr.shape[0] / 2)
    y = int(trim_repr.shape[1] / 2)

    # if isImage:
        # print(f'X is {x}')
        # print(f'Y is {y}')



    full_frs = torch.empty((full_rep.shape[0], x, y, 3), dtype=torch.double)
    if isImage:
        
        # ch = torch.norm(trim_repr.type(dtype=torch.double)).item()
        # print(ch, 'before')

        trim_repr = torch.from_numpy(cv2.resize(trim_repr.cpu().numpy(), (x, y),
                                                interpolation=cv2.INTER_AREA)).cuda()

        # ch = torch.norm(trim_repr.type(dtype=torch.double)).item()
        # print(ch, 'afyer')


        for i in range(0, total_fr):
            full_frs[i] = torch.from_numpy(cv2.resize(full_rep[i].cpu().numpy(), (x, y), interpolation=cv2.INTER_AREA)).cuda()

        full_rep = full_frs

    
    for i in range(0, total_fr):
        if (i == ref_fr_i) and (weight != 0.0 and weight != 1.0):
            start_fr = full_rep[ref_fr_i]
            end_fr = full_rep[ref_fr_i + 1]
            ref_fr = torch.mul(start_fr, weight) + torch.mul(end_fr, 1 - weight)
            sim_intrv = calc_sim(ref_fr, trim_repr)
            sim_arr.append(calc_sim(full_rep[i], trim_repr))
            sim_arr.append(sim_intrv)
        else:
            similarity = calc_sim(full_rep[i], trim_repr)
            sim_arr.append(similarity)
    return sim_arr


def consistency_loss(full_rep, trim_rep, full_span, trim_span, full, trim):
    try:
        # full = full.type(torch.double)
        # trim = trim.type(torch.double)
        
        #full = full.permute(1,2,0)
        # print(f'Before mean {trim_rep.shape}')
        trim_rep = torch.mean(trim_rep,dim=0) # TEMPORAL POOLING OF TRIM REP
        # print(f'After mean {trim_rep.shape}')


        ref_fr_no = full_rep.shape[0]
        fr_no = len(full_span)
        interval = int(fr_no / ref_fr_no)
        cn = int(len(trim_span)/2)
        
        ch = torch.norm(trim[cn].type(dtype=torch.double)).item()

        # if ch == 0:
        #     print('before processing trim zero')
        #     print(trim[cn])

        trim_cn= trim_span[cn]

        fr_int = []
        for i in range(0, fr_no, interval):
            fr_int.append(sum(full_span[i:(i + interval)]) / interval)

        if len(fr_int) > ref_fr_no:
            fr_int[ref_fr_no-1] = sum(fr_int[(ref_fr_no-1):])/(len(fr_int)- ref_fr_no + 1)

        fr_int = fr_int[0:ref_fr_no]

        full_samp = torch.empty((ref_fr_no, full.shape[1], full.shape[2], full.shape[3]), dtype=torch.double)
        # print('full_samp', full_samp.size())
        # print('frmae np', ref_fr_no)
        for i in range(0, ref_fr_no):
            # if (interval % 2) == 0:
            #     cnt = int((i * interval + (i + 1) * interval) / 2)
            #     full_samp[i] = torch.div((full[cnt - 1] + full[cnt]), 2.0)
            # else:
            cnt = int((i * interval + (i + 1) * interval) / 2)
            full_samp[i] = full[cnt]
            # cc = torch.norm(full_samp[i].type(dtype=torch.double)).item()
            #print('fullsamp', torch.norm(full_samp[i].type(dtype=torch.double)).item())
            # if cc==0:
            #     print('sabaina')
            # print('min, max', min(full_samp[i]), max(full_samp[i]))
        # print(full_samp.shape)
        # print(fr_int)
        # print(trim_cn)
        weight, ref_fr_i = find_interv_weight(fr_int, trim_cn)
        pred_trend = calc_sim_trend(full_rep, trim_rep, weight, ref_fr_i, False)
        act_trend = calc_sim_trend(full_samp, trim[cn], weight, ref_fr_i, True)

        # normalization of ground truth trend  (optional)
        act_trend = [(act_trend[i] - min(act_trend)) / (max(act_trend) - min(act_trend))
                    for i in range(0, len(act_trend))]

        a = 1 - (calc_sim(torch.from_numpy(np.array(pred_trend)),
                    torch.from_numpy(np.array(act_trend))))

        b = calc_loss(torch.from_numpy(np.array(pred_trend)),
                    torch.from_numpy(np.array(act_trend)))

        w = 0.8

        semisup_loss = w*a + b*(1-w)
        '''plt.plot(act_trend)
        #plt.plot(pred_trend)
        plt.legend(['act', 'pred'], loc='best')
        plt.show()'''
        return semisup_loss
    except:
        print('Semi-Loss Zero!')
        return 0


def calc_consistency_loss(full_rep, trim_rep, full_span, trim_span, full, trim):

    try:
        full = full.permute(0, 2, 3, 4,  1)
        trim = trim.permute(0, 2, 3, 4, 1)
        loss = 0
        batch_size = full.shape[0]
        for i in range(0, batch_size):
            loss += consistency_loss(full_rep[i], trim_rep[i],
                                    full_span[i], trim_span[i],
                                    full[i], trim[i])
        loss = np.array([loss])
        semisup_loss = torch.autograd.Variable(torch.from_numpy(loss), requires_grad=True)

        return semisup_loss
    except:
        return torch.Tensor([0])
        
if __name__ == '__main__':

    global train_list
    sup_w, self_w, semi_w = ramp_weight(50000)

    label = pickle.load(open('label.pkl','rb'))
    lu_bit = pickle.load(open('lu_bit.pkl','rb'))
    full_clip = pickle.load(open('full_clip.pkl','rb'))
    temporal_span_full = pickle.load(open('temporal_span_full.pkl','rb'))
    temporal_span_trimmed = pickle.load(open('temporal_span_trimmed.pkl','rb'))
    trimmed_clip = pickle.load(open('trimmed_clip.pkl','rb'))
    vid_paths = pickle.load(open('vid_paths.pkl', 'rb'))

    sup_ind = [lu_bit[i]*i  for i in range(0, len(lu_bit)) if lu_bit[i] is not 0 ]
    superv_labels = [label[i] for i in sup_ind]

    i = 0
    batch_size = full_clip.shape[0]
    criterion = nn.CrossEntropyLoss()
    train_list = np.load('train.list.npy')
    # pretxt_labels_full = torch.tensor([1, 0])
    pretxt_labels_full = torch.tensor([1])
    pretxt_labels_full = pretxt_labels_full.repeat(batch_size)

    # pretxt_labels_trim = torch.tensor([0, 1])
    pretxt_labels_trim = torch.tensor([0], dtype=torch.long)
    pretxt_labels_trim = pretxt_labels_trim.repeat(batch_size)

    # have to replace this by softmax output from pretext softmax layer for both videos
    pretxt_softmax_full = torch.rand((batch_size, 2))
    pretxt_softmax_trim = torch.rand((batch_size, 2))

    superv_softmax_full = torch.rand((len(sup_ind), 101))
    superv_softmax_trim = torch.rand((len(sup_ind), 101))


    #superv_labels = torch.randint(0, 100, (batch_size,), dtype=torch.long)


    self_a = criterion(pretxt_softmax_full, pretxt_labels_full)
    self_b = criterion(pretxt_softmax_trim, pretxt_labels_trim)

    self_sup_loss = (self_a + self_b)/2

    speed_weight = 0.8
    superv_labels = torch.from_numpy(np.array(superv_labels))
    sup_loss_a = criterion(superv_softmax_full, superv_labels)
    sup_loss_b = criterion(superv_softmax_trim, superv_labels)

    sup_loss = speed_weight*sup_loss_a + (1 - speed_weight)*sup_loss_b

    full_rep = 10*np.random.rand(batch_size, 6, 256, 34, 24)
    trim_rep = 5.70*np.random.rand(batch_size, 256, 34, 24)
    full_rep = torch.from_numpy(full_rep).type(torch.double)
    trim_rep = torch.from_numpy(trim_rep).type(torch.double)
    #full_rep = torch.rand(batch_size, 4, 256, 34, 24)
    #trim_rep = torch.rand(batch_size, 256, 34, 24)
    trim_cn = torch.randint(200, 1000, (batch_size,), dtype=torch.int)


    semi_sup_loss = calc_consistency_loss(full_rep, trim_rep,
                                        temporal_span_full, temporal_span_trimmed,
                                        full_clip, trimmed_clip)

    loss = torch.mul(self_sup_loss, self_w[i]) + \
        torch.mul(sup_loss, sup_w[i]) + \
        torch.mul(semi_sup_loss, semi_w[i])


#loss.backward()

