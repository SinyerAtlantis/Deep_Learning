# 基本深度学习模型

基于mxnet和gluon实现深度学习常用基本模型

代码主要来自李沐博士与Aston博士开设的[动手学深度学习][2]课程以及mxnet官方[tutorial][19]

由衷感谢沐神和mxnet小伙伴们的无私奉献！

以下notebook仅需要按照沐神课程逐步安装gpu版的mxnet及其依赖便可直接运行

1. from_strat：从零实现一个简单的图像分类器

    - Gradient：梯度下降法(Gradient descend)

      用numpy实现，利用梯度下降法求解高维线性分布的数据集的权重偏置

    - Gradient_Batch：随机梯度下降法(Stochastic gradient descend)

    - KNN：k=1最近邻分类cifar10数据集

    - MLP：多层感知机

      用numpy以及autograd自动求导实现cifar10数据集的分类

    - LeNet：用numpy及autograd实现LeNet分类cifar10数据集

2. cnn_cifar10

    通过gluon逐步构建复杂的卷积神经网络实现对cifar10的高精度分类(单模型95以上)，详细参数以及精度对比见文件夹内README.md

    代码参考了Wei Li通过keras的实现[BIGBALLON/cifar-10-cnn][1]

    cifar10数据集在官网自行下载解压后将load_cifar函数的route参数改为存放解压文件的绝对路径即可，下载地址：[CIFAR-10 python version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

    - mlp：多层感知机

    - lenet：调整了结构的lenet

    - lenet(data augmentation)：标准数据增强后的lenet

    - resnet50：前置batch normalization和relu的resnet50

    - resnet50(data augmentation)：加上标准数据集增强的resnet50

    - wide resnet(16\*8)：wrn实现高精度分类

    - kaggle(cifar10)：用wrn16\*8模型参加kaggle cifar10比赛

      单模型精度95.96，ensemble后精度96.98。单模型精度即可击败原比赛榜单第一

    - kaggle(house price)：房价预测，沐神课程初期的一个小练习

      由于同属于kaggle比赛，且训练很快，可以随手跑一跑，代码为一个简单调参后的demo，精度应在0.117左右，名次约在16%，传统机器学习方法应该可以获得更好的成绩，详细可参考[实战Kaggle比赛——使用Gluon预测房价和K折交叉验证](http://zh.gluon.ai/chapter_supervised-learning/kaggle-gluon-kfold.html)

3. transfer_learning：迁移学习

    - prediction：利用预训练模型直接完成分类任务（待补）

    - fine tuning：微调预训练模型完成热狗分类

    - neural style：样式迁移

      基于VGG19预训练模型实现图片风格样式迁移

    - kaggle(dog breed identification)：kaggle120种狗分类比赛

      基于inception v3和resnet152 v1预训练模型通过迁移学习训练模型分类120种狗，使用原始数据集精度可达0.2673，使用stanford数据集精度可达0.0038，具体细节见文件夹内README.md

    PS：网络环境不好时可复制下载链接（注意去掉最后三个点）丢到迅雷里面下载，然后把下载好的模型参数拷到/.mxnet/model文件夹下。

4. gan（draft）

    generative adversarial networks / 生成对抗网络

    - conv_gan

      使用dcgan(deep convolutional gan)生成人脸

    - condition_gan

      使用conditional gan生成mnist数字

5. detection（draft）

    object detection and semantic segmentation / 目标检测与语义分割

    - faster rcnn
    - ssd：Single Shot MultiBox Detector
    - yolo：You Only Look Once: Unified, Real-Time Object Detection
    - fcn：Fully Convolutional Networks for Semantic Segmentation

6. rnn：基本循环神经网络（draft）

    - rnn_base

7. rl：reinforcement learning（draft）

    - dqn

    - ddqn

相关论文：

1. [Gradient-Based Learning Applied to Document Recognition][8] (Lenet)
2. [Deep Residual Learning for Image Recognition][3] (Resnet)
3. [Identity Mappings in Deep Residual Networks][4] (Resnet_v2)
4. [Wide Residual Networks][5]
5. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks][11]
6. [SSD: Single Shot MultiBox Detector][7]
7. [You Only Look Once: Unified, Real-Time Object Detection][20]
8. [Fully Convolutional Networks for Semantic Segmentation][21]
9. [Generative Adversarial Networks][15]
10. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][24]
11. [Conditional Generative Adversarial Nets][14]
12. [Deep Reinforcement Learning: An Overview][13]
13. [Human-level control through deep reinforcement learning][23]

扩展阅读：

1. [ImageNet Classification with Deep Convolutional Neural Networks][9] (Alexnet)
2. [Going Deeper with Convolutions][16] (Googlenet)
3. [Very Deep Convolutional Networks for Large-Scale Image Recognition][17] (VGG)
4. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift][6]
5. [Rich feature hierarchies for accurate object detection and semantic segmentation][10] (RCNN)
6. [Fast R-CNN][18]
7. [Mask R-CNN][12]
8. [Image-to-Image Translation with Conditional Adversarial Networks][22]

[1]: https://github.com/BIGBALLON/cifar-10-cnn
[2]: https://www.bilibili.com/video/av14327359/?from=search&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;seid=4696511599201035761
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
[16]: https://arxiv.org/abs/1409.4842
[17]: https://arxiv.org/abs/1409.1556
[18]: https://arxiv.org/abs/1504.08083
[19]: https://github.com/zackchase/mxnet-the-straight-dope
[20]: https://arxiv.org/abs/1506.02640
[21]: https://arxiv.org/abs/1411.4038
[22]: https://arxiv.org/abs/1611.07004
[23]: https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/
[24]: https://arxiv.org/abs/1511.06434