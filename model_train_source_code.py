import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LeakyReLU, PReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dropout
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
from random import shuffle
from keras.models import load_model
from keras import backend as K


def subsample_clutter(x, y, sampleratio):
    clutter_ind = np.where(y == 11)
    clutter_ind = np.array(clutter_ind)
    clutter_ind = clutter_ind.reshape((-1, 1))
    np.random.shuffle(clutter_ind)
    clutter_size = clutter_ind.shape[0]
    cl_discard = (np.ceil(np.ceil(clutter_size * sampleratio))).astype('int32')
    discard_ind = clutter_ind[0:cl_discard]
    y_new = np.delete(y, discard_ind, 0)
    x_new = np.delete(x, discard_ind, 0)
    return x_new, y_new


def _shuffle(x, y):
    samplesNo = y.shape[0]
    shuffle_x = []
    shuffle_y = []
    index_shuf = list(range(samplesNo))
    shuffle(index_shuf)
    for i in index_shuf:
        shuffle_x.append(x[i])
        shuffle_y.append(y[i])

    shuffle_x = np.asarray(shuffle_x)
    shuffle_y = np.asarray(shuffle_y)

    return shuffle_x, shuffle_y


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

y_test_day_3_5 = np.load('y_test_day_3_5.npy')
x_test_day_3_5 = np.load('x_test_day_3_5.npy')

x_train_day_2_5 = np.load('x_train_day_2_5.npy')
y_train_day_2_5 = np.load('y_train_day_2_5.npy')

x_train_night_2_5 = np.load('x_train_night_2_5.npy')
y_train_night_2_5 = np.load('y_train_night_2_5.npy')

x_tr = []
y_tr = []

day_imgs_35 = y_test_day_3_5.shape[0]
day_imgs = y_train_day_2_5.shape[0]
night_imgs = y_train_night_2_5.shape[0]

for i in range(0, day_imgs_35):
    x_tr.append(x_test_day_3_5[i])
    y_tr.append(y_test_day_3_5[i])

for i in range(0, day_imgs):
    x_tr.append(x_train_day_2_5[i])
    y_tr.append(y_train_day_2_5[i])
#
for i in range(0, night_imgs):
    x_tr.append(x_train_night_2_5[i])
    y_tr.append(y_train_night_2_5[i])

x_tr = np.array(x_tr)
y_tr = np.array(y_tr)

samplesNo = y_tr.shape[0]

shuffle_x = []
shuffle_y = []
index_shuf = list(range(samplesNo))
shuffle(index_shuf)
for i in index_shuf:
    shuffle_x.append(x_tr[i])
    shuffle_y.append(y_tr[i])

shuffle_x = np.asarray(shuffle_x)
shuffle_y = np.asarray(shuffle_y)

a = np.ceil(0.3*samplesNo)
a = a.astype('int32')
x_train = shuffle_x[0:samplesNo-a]
x_test = shuffle_x[samplesNo-a:]

y_train = shuffle_y[0:samplesNo-a]
y_test = shuffle_y[samplesNo-a:]


x_train, y_train = subsample_clutter(x_train, y_train, 0.7)
x_test, y_test = subsample_clutter(x_test, y_test, 0.95)


num_classes = 10
batch_size = 128

x_train = x_train
y_train = y_train

train_no, width_tr, height_tr = x_train.shape[:3]
test_no, width_test, height_test = x_test.shape[:3]

x_train = x_train.astype('float32')
# zero centre mean normalization
for i in range(0, train_no):
    sum_ = x_train[i, :, :].sum()
    totalpixels = width_tr * height_tr * 3
    mean = sum_ / totalpixels
    x_train[i, :, :] -= mean
x_train /= 255

x_test = x_test.astype('float32')
for i in range(0, test_no):
    sum_ = x_test[i, :, :].sum()
    totalpixels = width_test * height_test * 3
    mean = sum_ / totalpixels
    x_test[i, :, :] -= mean
x_test /= 255

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

clutter_ind = np.where(y_train == 11)

y_train -= 2
y_test -= 2

# Convert class vectors to binary class matrices. One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

class_weights = {0: 0.99, 1: 0.99, 2: 0.99, 3: 0.99, 4: 0.99, 5: 0.99, 6: 0.99, 7: 0.99, 8: 0.99, 9: 0.99, 10: 0.01}

model = Sequential()
model.add(
    Conv2D(64, (9, 11), kernel_initializer=keras.initializers.he_normal(seed=None), input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (9, 11), kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Conv2D(128, (7, 9), kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(Conv2D(128, (7, 9), kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 7), padding='same', kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(Conv2D(256, (5, 7), padding='same', kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 5), padding='same', kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(Conv2D(512, (3, 5), padding='same', kernel_initializer=keras.initializers.he_normal(seed=None)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

print(model.summary())
# Train model

adam = keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
sgd = keras.optimizers.sgd(lr=1e-4, decay=1e-3, momentum=0.955, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy', f1_m, precision_m, recall_m])

# fit the model

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test),
                    shuffle=True, verbose=2, class_weight=class_weights)

# model.save('atr_model.h5')


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['f1_m'])
plt.plot(history.history['val_f1_m'])
plt.plot(history.history['precision_m'])
plt.plot(history.history['val_precision_m'])
plt.plot(history.history['recall_m'])
plt.plot(history.history['val_recall_m'])
plt.title('model evaluation')
#plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

plt.legend(['train accuracy', 'test accuracy', 'train f1', 'test f1', 'train precision', 'test precision'
            , 'train recall', 'test recall'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)

y_pred = model.predict(x_test, verbose=2)
confusion_matrix = np.zeros((num_classes, num_classes))

actual = np.argmax(y_test, axis=1)
pred = np.argmax(y_pred, axis=1)

for i in range(0, len(y_test)):
    confusion_matrix[actual[i], pred[i]] = confusion_matrix[actual[i], pred[i]] + 1

# find accuracy

diagnol_sum = confusion_matrix.trace()
total_elements = confusion_matrix.sum()
confusion_matrix = confusion_matrix.astype('int')
accuracy = diagnol_sum / total_elements
print(accuracy, 'accuracy')
print(confusion_matrix, 'confusion_matrix')

# print class errors
for i in range(0, num_classes):
    class_metric = confusion_matrix[i, :]
    class_accuracy = confusion_matrix[i, i] / class_metric.sum()
    print('accuracy for class ' + str(i + 1) + ' is {:0.2f}%'.format((class_accuracy) * 100))

plt.figure(1), plt.imshow(confusion_matrix), plt.title('confusion_matrix')
plt.show()

################# testing on different time and range ########################
# y_test_day_3_5 = np.load('y_test_day_3_5.npy')
# x_test_day_3_5 = np.load('x_test_day_3_5.npy')
#
# # x_test_shuffle, y_test_shuffle = _shuffle(x_test, y_test)
# #
# # samplesNo = y_test.shape[0]
# #
# # a = np.ceil(0.5 * samplesNo)
# # a = a.astype('int32')
# # x_tr_ = x_test_shuffle[0:samplesNo - a]
# # y_tr_ = y_test_shuffle[0:samplesNo - a]
# #
# # x_test = x_test_shuffle[samplesNo - a:]
# # y_test = y_test_shuffle[samplesNo - a:]
#
# x_train_day_2_5 = np.load('x_train_day_2_5.npy')
# y_train_day_2_5 = np.load('y_train_day_2_5.npy')
#
# x_train_night_2_5 = np.load('x_train_night_2_5.npy')
# y_train_night_2_5 = np.load('y_train_night_2_5.npy')
#
# # x_tr = []
# # y_tr = []
# #
# # night_imgs = y_train_night_2_5.shape[0]
# # day_imgs = y_train_day_2_5.shape[0]
# #
# # for i in range(0, night_imgs):
# #     x_tr.append(x_train_night_2_5[i])
# #     y_tr.append(y_train_night_2_5[i])
# #
# # for i in range(0, day_imgs):
# #     x_tr.append(x_train_day_2_5[i])
# #     y_tr.append(y_train_day_2_5[i])
# # #
# # # for i in range(0, y_tr_.shape[0]):
# # #     x_tr.append(x_tr_[i])
# # #     y_tr.append(y_tr_[i])
# #
# # x_train = np.array(x_tr)
# # y_train = np.array(y_tr)
# #
# # x_train, y_train = subsample_clutter(x_train, y_train, 0.7)
# # x_test, y_test = subsample_clutter(x_test, y_test, 0.95)
#
# x_test_shuffle, y_test_shuffle = _shuffle(x_train_night_2_5, y_train_night_2_5)
#
# samplesNo = y_train_night_2_5.shape[0]
#
# a = np.ceil(0.5*samplesNo)
# a = a.astype('int32')
# x_tr_ = x_test_shuffle[0:samplesNo-a]
# y_tr_ = y_test_shuffle[0:samplesNo-a]
#
#
# x_test = x_test_shuffle[samplesNo-a:]
# y_test = y_test_shuffle[samplesNo-a:]
#
#
# x_tr = []
# y_tr = []
#
# day_imgs_35 = y_test_day_3_5.shape[0]
# day_imgs = y_train_day_2_5.shape[0]
#
# for i in range(0, day_imgs_35):
#     x_tr.append(x_test_day_3_5[i])
#     y_tr.append(y_test_day_3_5[i])
#
# for i in range(0, day_imgs):
#     x_tr.append(x_train_day_2_5[i])
#     y_tr.append(y_train_day_2_5[i])
# #
# for i in range(0, y_tr_.shape[0]):
#     x_tr.append(x_tr_[i])
#     y_tr.append(y_tr_[i])
#
# x_train = np.array(x_tr)
# y_train = np.array(y_tr)