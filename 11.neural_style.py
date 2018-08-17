import mxnet as mx
from mxnet import image, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 150

ctx = mx.gpu()

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std + rgb_mean).clip(0,1)

def extract_features(x, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        x = net[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles

def get_net(pretrained_net, content_layers, style_layers):
    net = nn.Sequential()
    for i in range(max(content_layers+style_layers)+1):
        net.add(pretrained_net.features[i])
    return net

def content_loss(yhat, y):
    return (yhat-y).square().mean()

def gram(x):
    c = x.shape[1]
    n = x.size / x.shape[1]
    y = x.reshape((c, int(n)))
    return nd.dot(y, y.T) / n

def style_loss(yhat, gram_y):
    return (gram(yhat) - gram_y).square().mean()

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean()+(yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())

def sum_loss(loss, preds, truths, weights):
    return nd.add_n(*[w*loss(yhat, y) for w, yhat, y in zip(weights, preds, truths)])

def get_contents(image_shape):
    content_x = preprocess(content_img, image_shape).copyto(ctx)
    content_y, _ = extract_features(content_x, content_layers, style_layers)
    return content_x, content_y

def get_styles(image_shape):
    style_x = preprocess(style_img, image_shape).copyto(ctx)
    _, style_y = extract_features(style_x, content_layers, style_layers)
    style_y = [gram(y) for y in style_y]
    return style_x, style_y

style_img = image.imread('style1.jpg')
content_img = image.imread('content.jpg')

plt.imshow(style_img.asnumpy())
plt.show()
plt.imshow(content_img.asnumpy())
plt.show()

pretrained_net = models.vgg19(pretrained=True)
style_layers = [0,5,10,19,28]
content_layers = [25]
net = get_net(pretrained_net, content_layers, style_layers)

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

channels = [net[l].weight.shape[0] for l in style_layers]
style_weights = [1e4/n**2 for n in channels]
content_weights = [1]
tv_weight = 10

a = len(content_img[0])
image_shape = (1440, int(len(content_img)*(1440/len(content_img[0]))))

net.collect_params().reset_ctx(ctx)

content_x, content_y = get_contents(image_shape)
style_x, style_y = get_styles(image_shape)

x = content_x.copyto(ctx)
x.attach_grad()

lr = 0.1
epochs = 300

start = time()
for i in range(epochs):
    with autograd.record():
        content_py, style_py = extract_features(x, content_layers, style_layers)
        content_l  = sum_loss(content_loss, content_py, content_y, content_weights)
        style_l = sum_loss(style_loss, style_py, style_y, style_weights)
        tv_l = tv_weight * tv_loss(x)
        loss = style_l + content_l + tv_l
    loss.backward()
    x.grad[:] /= x.grad.abs().mean()+1e-8
    x[:] -= lr * x.grad
    nd.waitall()
    if i % 20 == 0:
        print('E %d, content %.2f, style %.2f, TV %.2f, T %.2f' % (
            i,content_l.asscalar(),style_l.asscalar(), tv_l.asscalar(), time()-start))
        start = time()
    if i in [40, 160, 240]:
        lr = lr * 0.1

out = postprocess(x).asnumpy()
plt.imshow(out)
plt.show()
plt.imsave('out.png', out)