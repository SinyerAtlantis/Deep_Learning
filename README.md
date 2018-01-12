# 深度学习

基于mxnet和gluon逐步实现深度学习常用模型

代码参考了沐神[动手学深度学习][2]课程，以及Wei Li通过keras的实现[BIGBALLON/cifar-10-cnn][1]。沐神辛苦开设的高质量深度学习课程和Wei Li高质量的keras代码给了我很多启发，大大加速了我对深度学习的学习，由衷感谢！

以下notebook仅需要按照沐神课程逐步安装gpu版的mxnet及其依赖便可直接运行。cifar10数据集在官网自行下载解压后将load_cifar函数的route参数改为存放解压文件的绝对路径即可，下载地址如下：

[CIFAR-10 python version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

1. from_strat

    从零逐步实现图像分类器，

    - Gradient：梯度下降法(Gradient descend)

      用numpy实现，利用梯度下降法求解高维线性分布的数据集的权重偏置

    - Gradient_Batch：随机梯度下降法(Stochastic gradient descend)

    - KNN：k=1最近邻分类cifar10数据集

    - MLP：多层感知机

      用numpy以及autograd自动求导实现cifar10数据集的分类

    - LeNet：用numpy及autograd实现LeNet分类cifar10数据集

2. cifar10_gluon

    通过gluon逐步构建复杂的卷积神经网络实现对cifar10的高精度分类，详细参数见文件夹内README.md。

    - mlp：多层感知机
    - lenet：调整了一点结构的lenet
    - lenet(data augmentation)：标准数据增强后的lenet
    - resnet50：前置batch normalization和relu的resnet50
    - resnet50(data augmentation)：加上标准数据集增强
    - wide resnet(16*8)：wrn实现高精度分类
    - kaggle(wrn)：用于kaggle cifar10比赛用的脚本

3. detection

    正在制作中

    - ssd：Single Shot MultiBox Detector
    - mask rcnn

4. image_caption

    正在制作中

5. GAN

    规划中

6. Reinforcement learning

    规划中

keras的个人学习用notebook正在制作中，参考至[BIGBALLON/cifar-10-cnn][1]，更新速度较慢。

相关论文：

1. [Gradient-Based Learning Applied to Document Recognition][8]
2. [Deep Residual Learning for Image Recognition][3]
3. [Identity Mappings in Deep Residual Networks][4]
4. [Wide Residual Networks][5]
5. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift][6]
6. [SSD: Single Shot MultiBox Detector][7]
7. [Mask R-CNN][12]
8. [Deep Reinforcement Learning: An Overview][13]
9. [Generative Adversarial Networks][15]

扩展阅读：

1. [ImageNet Classification with Deep Convolutional Neural Networks][9] (Alexnet)
2. [Rich feature hierarchies for accurate object detection and semantic segmentation][10] (RCNN)
3. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks][11]
4. [Conditional Generative Adversarial Nets][14]

[1]: https://github.com/BIGBALLON/cifar-10-cnn
[2]: https://www.bilibili.com/video/av14327359/?from=search&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;seid=4696511599201035761
[3]: https://arxiv.org/abs/1512.03385
[4]: https://arxiv.org/abs/1603.05027
[5]: https://arxiv.org/abs/1605.07146
[6]: https://arxiv.org/abs/1502.03167
[7]: https://arxiv.org/abs/1512.02325
[8]: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
[9]: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
[10]: https://arxiv.org/abs/1311.2524
[11]: https://arxiv.org/abs/1506.01497
[12]: https://arxiv.org/abs/1703.06870
[13]: https://arxiv.org/abs/1701.07274
[14]: https://arxiv.org/abs/1411.1784
[15]: https://arxiv.org/abs/1406.2661