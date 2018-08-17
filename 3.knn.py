import gzip
import struct
import numpy as np
data_dir = r'C:\Users\sinyer\Desktop\mnist'


def load_mnist():
    def read_data(label_dir, image_dir):
        with gzip.open(label_dir) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(image_dir, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        return label, image
    train_label, train_img = read_data(
        data_dir+'/train-labels-idx1-ubyte.gz', data_dir+'/train-images-idx3-ubyte.gz')
    test_label, test_img = read_data(
        data_dir+'/t10k-labels-idx1-ubyte.gz', data_dir+'/t10k-images-idx3-ubyte.gz')
    return train_img, train_label, test_img, test_label


def loss(train_data, test_data, i):
    temp_list = []
    distance = np.abs(train_data - test_data[i, :])
    for j in range(len(distance)):
        temp_list.append(np.sum(distance[j]))
    return temp_list


train_img, train_label, test_img, test_label = load_mnist()

train_data = train_img[:5000].astype(np.float16)
test_data = test_img[:100].astype(np.float16)
train_label = train_label[:5000]
test_label = test_label[:100]


num = len(test_data)
pred_label = np.zeros(num)

for i in range(num):
    index = np.argmin(loss(train_data, test_data, i))
    pred_label[i] = train_label[index]

acc = np.mean(pred_label == test_label)
print(acc)
