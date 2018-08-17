import mxnet as mx
from mxnet import init, gluon, nd, autograd, image
from mxnet.gluon.model_zoo import vision
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from time import time
ctx = mx.gpu()
data_dir = '/home/sinyer/python/data/hotdog'

def load_data(route = data_dir):
    train_imgs = gluon.data.vision.ImageFolderDataset(route+'/train',
        transform=lambda X, y: augment(X, y, train_augs))
    test_imgs = gluon.data.vision.ImageFolderDataset(route+'/test',
        transform=lambda X, y: augment(X, y, test_augs))
    return train_imgs, test_imgs

def augment(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx):
    acc = 0.
    for data, label in data_iterator:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

batch_size = 32
train_augs = [image.HorizontalFlipAug(.5),image.RandomCropAug((224,224))]
test_augs = [image.CenterCropAug((224,224))]

train_imgs, test_imgs = load_data()
train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_imgs, batch_size)

pretrained_net = vision.resnet18_v2(pretrained=True)

net = vision.resnet18_v2(classes=2)
net.features = pretrained_net.features
net.output.initialize(init.Xavier())

net.collect_params().reset_ctx(ctx)
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-3})

epochs = 10

a = []
b = []
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    start = time()
    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            l = loss(output, label)
        l.backward()
        trainer.step(batch_size)
        train_loss = train_loss + nd.mean(l).asscalar()
        train_acc = train_acc + accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net, ctx)

    if epoch % 1 == 0:
        print(epoch, 'loss:%.4f tracc:%.4f teacc:%.4f time:%.3f' % (
            train_loss / len(train_data), train_acc / len(train_data), test_acc, time() - start))
    a.append(train_acc / len(train_data))
    b.append(test_acc)

print('tracc:%f teacc:%f' % (train_acc / len(train_data), test_acc))
plt.plot(np.arange(epochs), a, np.arange(epochs), b)
plt.ylim(0, 1)
plt.show()
