import mxnet as mx
from mxnet import gluon, nd, autograd, image
from mxnet.gluon import nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from time import time
ctx = mx.gpu()
data_dir = '/home/sinyer/python/data/cifar10/cifar-10-batches-py'


def load_cifar(route):
    def load_batch(filename):
        with open(filename, 'rb')as f:
            data = pickle.load(f, encoding='latin1')
            pic = data['data']
            label = data['labels']
            pic = pic.reshape(10000, 3, 32, 32).astype("float")
            label = np.array(label)
            return pic, label

    x1, y1 = load_batch(route + "/data_batch_1")
    x2, y2 = load_batch(route + "/data_batch_2")
    x3, y3 = load_batch(route + "/data_batch_3")
    x4, y4 = load_batch(route + "/data_batch_4")
    x5, y5 = load_batch(route + "/data_batch_5")
    test_pic, test_label = load_batch(route + "/test_batch")
    train_pic = np.concatenate((x1, x2, x3, x4, x5))
    train_label = np.concatenate((y1, y2, y3, y4, y5))
    return train_pic, train_label, test_pic, test_label


def evaluate_accuracy(data_iterator, net, ctx):
    acc = 0.
    for data, label in data_iterator:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        acc += nd.mean(output.argmax(axis=1) == label).asscalar()
    return acc / len(data_iterator)


def augment(data):
    aug_list = image.CreateAugmenter(data_shape=(3, 32, 32), rand_crop=True, rand_mirror=True)
    data = nd.pad(data, pad_width=(0, 0, 0, 0, 2, 2, 2, 2), mode='constant', constant_value=0)
    data = nd.transpose(data, (0, 2, 3, 1))
    temp = []
    for d in data:
        for aug in aug_list:
            d = aug(d)
        temp.append(d)
    data = nd.stack(*temp)
    data = nd.transpose(data, (0, 3, 1, 2))
    return data


train_pic, train_label, test_pic, test_label = load_cifar(data_dir)

# ------------------------------------ #

train_pic = np.transpose(train_pic.astype('float32')/255, (0, 2, 3, 1))
test_pic = np.transpose(test_pic.astype('float32')/255, (0, 2, 3, 1))
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
for i in range(3):
    train_pic[:, :, :, i] = (train_pic[:, :, :, i] - mean[i])/std[i]
    test_pic[:, :, :, i] = (test_pic[:, :, :, i] - mean[i])/std[i]
train_pic = np.transpose(train_pic, (0, 3, 1, 2))
test_pic = np.transpose(test_pic, (0, 3, 1, 2))

aug_train = image.CreateAugmenter(data_shape=(3, 32, 32), rand_crop=True, rand_mirror=True)

# ------------------------------------ #

batch_size = 128
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_pic, train_label.astype('float32')),
                                   batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_pic, test_label.astype('float32')),
                                  batch_size, shuffle=False)

# ------------------------------------ #

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
net.initialize(ctx=ctx)

loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'nag', {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 5e-4})

# ------------------------------------ #

epochs = 110

a = []
b = []
for epoch in range(epochs):
    if epoch == 50:
        trainer.set_learning_rate(0.01)
    if epoch == 80:
        trainer.set_learning_rate(0.001)
    train_loss = 0.
    train_acc = 0.
    start_time = time()

    for data, label in train_data:
        data = augment(data).as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = loss_function(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += nd.mean(output.argmax(axis=1) == label).asscalar()
    test_acc = evaluate_accuracy(test_data, net, ctx)

    if epoch % 10 == 0:
        print('epoch: %d, train acc: %.4f, test acc: %.4f, epoch time: %.3f' % (
            epoch, train_acc / len(train_data), test_acc, time() - start_time))
    a.append(train_acc / len(train_data))
    b.append(test_acc)

print('train acc: %f, test acc: %f' % (train_acc / len(train_data), test_acc))
plt.plot(np.arange(epochs), a, np.arange(epochs), b)
plt.ylim(0, 1)
plt.show()
