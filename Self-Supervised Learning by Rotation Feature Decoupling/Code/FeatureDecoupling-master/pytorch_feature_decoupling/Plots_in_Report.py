import numpy as np
import matplotlib.pyplot as plt

conv1_tr = np.load('conv1_tr.npy')
conv2_tr = np.load('conv2_tr.npy')
conv3_tr = np.load('conv3_tr.npy')
conv4_tr = np.load('conv4_tr.npy')
conv5_tr = np.load('conv5_tr.npy')
loss_tr = np.load('loss_tr.npy')

plt.figure(2)
plt.plot(conv1_tr)
plt.plot(conv2_tr)
plt.plot(conv3_tr)
plt.plot(conv4_tr)
plt.plot(conv5_tr)
plt.plot(loss_tr)
plt.legend(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'loss'])
plt.xlabel('Epochs')
plt.ylabel('Linear precision')
plt.show()

conv1_ts = np.load('conv1_ts.npy')
conv2_ts = np.load('conv2_ts.npy')
conv3_ts = np.load('conv3_ts.npy')
conv4_ts = np.load('conv4_ts.npy')
conv5_ts = np.load('conv5_ts.npy')
loss_ts = np.load('loss_ts.npy')

plt.figure(2)
plt.plot(conv1_ts)
plt.plot(conv2_ts)
plt.plot(conv3_ts)
plt.plot(conv4_ts)
plt.plot(conv5_ts)
plt.plot(loss_ts)
plt.legend(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'loss'])
plt.xlabel('Epochs')
plt.ylabel('Linear precision')
plt.show()

nconv1_tr = np.load('nconv1_tr.npy')
nconv2_tr = np.load('nconv2_tr.npy')
nconv3_tr = np.load('nconv1_ts.npy')
nconv4_tr = np.load('nconv2_ts.npy')
nloss_tr = np.load('nloss_tr.npy')
nloss_ts = np.load('nloss_ts.npy')

plt.figure(2)
plt.plot(nconv1_tr)
plt.plot(nconv2_tr)
plt.plot(nconv3_tr)
plt.plot(nconv4_tr)
plt.plot(nloss_tr )
plt.plot(nloss_ts )
plt.legend(['conv4 train', 'conv5 train', 'conv4 test', 'conv5 test', 'loss train', 'loss test'])
plt.xlabel('Epoch')
plt.ylabel('Non Linear Precision')
plt.show()

import math
import matplotlib.pyplot as plt
tinysizebyBatch = 100000.0/192
total_iterations= (200*tinysizebyBatch) + tinysizebyBatch
cls = []
mse = []
nce = []
ramp_up_end_slf = int(total_iterations/3)
ramp_up_end_sem = ramp_up_end_slf*2
for iteration in range(0, int(total_iterations)):
    if (iteration > ramp_up_end_slf) and (iteration < ramp_up_end_sem):
        ramp_weight = 1
    elif iteration > ramp_up_end_slf:
        ramp_weight = math.exp(-30 * math.pow((1 - ramp_up_end_sem / (iteration)), 2))
    else:
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end_slf), 2))
    cls.append(ramp_weight)
    if iteration < ramp_up_end_slf:
        ramp_weight = 1
    else:
        ramp_weight = math.exp(-20 * math.pow((1 - ramp_up_end_slf / (iteration)), 2))
    mse.append(ramp_weight)
    if iteration > ramp_up_end_sem:
        ramp_weight = 1
    else:
        ramp_weight = math.exp(-15 * math.pow((1 - iteration / ramp_up_end_sem), 2))
    nce.append(ramp_weight)

fig, axs = plt.subplots(3,1)
plt.title('ramp weights for losses')
axs[0].plot(mse), axs[0].set_title('Rot loss')
axs[1].plot(cls),  axs[1].set_title('MSE loss')
axs[2].plot(nce), axs[2].set_title('NCE loss')
#plt.legend(['supervised', 'self-supervised', 'semi-supervised'], loc='best')
for ax in axs.flat:
    ax.label_outer()
plt.show()

plt.figure(2)
plt.plot(cls)
plt.plot(mse)
plt.plot(nce)
plt.legend(['Rot loss', 'MSE loss', 'NCE loss'], loc='best')
plt.show()
