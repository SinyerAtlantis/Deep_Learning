import numpy as np
import matplotlib.pyplot as plt


x = np.random.random((100, 2))

init_w = np.array([[1.4], [0.9]])
init_b = 0.2

y = np.dot(x, init_w) + init_b  # y = wx + b

w = np.random.randn(2, 1)
b = 0

lr = 0.01
epochs = 200
batch = 10

a = []
for e in range(epochs):
    for i in range(int(len(x) / batch)):
        x_batch = x[batch * i: batch * (i + 1)]
        y_batch = y[batch * i: batch * (i + 1)]
        w = w - lr * 2 * np.dot(x_batch.T, np.dot(x_batch, w) + b - y_batch)
        b = b - lr * 2 * (np.dot(x_batch, w) + b - y_batch).sum()
    a.append(((np.dot(x, w) + b - y) ** 2).sum())

print(w, b, a[-1])
plt.plot(np.arange(epochs), a)
plt.show()
