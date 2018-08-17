import numpy as np
import matplotlib.pyplot as plt


x = np.random.random((100, 2))

init_w = np.array([[1.4], [0.9]])
init_b = 0.2

y = np.dot(x, init_w) + init_b  # y = wx + b

w = np.random.randn(2, 1)
b = 0

# loss = ((wx + b) - y) ** 2
# d(loss)/dw = 2(wx + b - y) * x
# d(loss)/db = 2(wx + b - y)
# w = w - lr * loss_gradient

learning_rate = 0.01
epochs = 100

a = []
for e in range(epochs):
    w = w - learning_rate * 2 * np.dot(x.T, (np.dot(x, w) + b - y))
    b = b - learning_rate * 2 * (np.dot(x, w) + b - y).sum()
    a.append(((np.dot(x, w) + b - y) ** 2).sum())

print(w, b, a[-1])
plt.plot(np.arange(epochs), a)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
