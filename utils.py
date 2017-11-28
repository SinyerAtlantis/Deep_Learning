from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import numpy as np
import pickle as p
from time import time
import matplotlib.pyplot as plt

def load_cifar(train_num, test_num, batch_size, route='../Data'):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb')as f:
            data_dict = p.load(f, encoding='latin1')
            X = data_dict['data']
            Y = data_dict['labels']
            X = X.reshape(10000, 3, 32,32).astype("float")
            Y = np.array(Y)
            return X, Y
    def load_CIFAR_Labels(filename):
        with open(filename, 'rb') as f:
            label_names = p.load(f, encoding='latin1')
            names = label_names['label_names']
            return names
    label_names = load_CIFAR_Labels(route + "/cifar-10-batches-py/batches.meta")
    img_X1, img_Y1 = load_CIFAR_batch(route + "/cifar-10-batches-py/data_batch_1")
    img_X2, img_Y2 = load_CIFAR_batch(route + "/cifar-10-batches-py/data_batch_2")
    img_X3, img_Y3 = load_CIFAR_batch(route + "/cifar-10-batches-py/data_batch_3")
    img_X4, img_Y4 = load_CIFAR_batch(route + "/cifar-10-batches-py/data_batch_4")
    img_X5, img_Y5 = load_CIFAR_batch(route + "/cifar-10-batches-py/data_batch_5")
    test_pic, test_label = load_CIFAR_batch(route + "/cifar-10-batches-py/test_batch")
    train_pic = np.concatenate((img_X1, img_X2, img_X3, img_X4, img_X5))
    train_label = np.concatenate((img_Y1, img_Y2, img_Y3, img_Y4, img_Y5))
    X = train_pic[:train_num,:].astype('float32')/255
    y = train_label[:train_num].astype('float32')
    X_ = test_pic[:test_num,:].astype('float32')/255
    y_ = test_label[:test_num].astype('float32')
    mean=np.array([0.4914, 0.4822, 0.4465])
    std=np.array([0.2023, 0.1994, 0.2010])
    for i in range(3):
        X[:,:,:,i] = (X[:,:,:,i] - mean[i]) / std[i]
        X_[:,:,:,i] = (X_[:,:,:,i] - mean[i]) / std[i]
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_, y_), batch_size, shuffle=False)
    return train_data, test_data

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

def train(ctx, train_data, test_data, net, loss, trainer, epochs, n=1, print_batches=None):
    a = []
    b = []
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        batch = 0
        start = time()
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()
            trainer.step(data.shape[0])
            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)
            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d, Loss: %f, Train acc %f" % (batch, train_loss/batch, train_acc/batch))
        a.append(train_acc/batch)
        test_acc = evaluate_accuracy(test_data, net, ctx)
        b.append(test_acc)
        if epoch%n == 0:
            print("Epoch %d, Loss: %f, Train acc %f, Test acc %f, Time %f" % (
                epoch, train_loss/batch, train_acc/batch, test_acc, time() - start))
    plt.plot(np.arange(0, epochs),a,np.arange(0, epochs),b)
    plt.ylim(0, 1)
    plt.show()