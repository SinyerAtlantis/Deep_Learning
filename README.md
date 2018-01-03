# Gluon

重置中

计划完成
1. from_strat
- 1. Gradient
- 2. KNN
- 3. MLP
- 4. LeNet

2. cifar10_gluon
- 1. MLP
- 2. LeNet
- 3. GoogleNet
- 4. ResNet 10
- 5. ResNet 50
- 6. ResNet 50(data_augmentation)

3. detection

4. image_caption

运行notebook需要cifar10官网下载的python数据集

由于国内网络条件不佳，建议不使用框架自带的load数据的程序，可先从官网下载下来并解压，下载完成后将notebook里的route参数更改为存放解压文件夹的路径即可

[CIFAR-10 python version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

注：使用了数据增强的模型(resnet_50_data_aug)由于其增强接口我还不太清晰，无法调用Python版的数据，故使用了官方以C为基础的cifar10数据集，同样可以自行下载后，更改gluon.data.vision.CIFAR10函数参数内的root参数为解压文件夹路径即可

[CIFAR-10 binary version (suitable for C programs)](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

keras的notebook正在制作中，主要参考[BIGBALLON/cifar-10-cnn][1]

 [1]: https://github.com/BIGBALLON/cifar-10-cnn
