# 分类网络

### 几篇综述
* [从VGG到NASNet，一文概览图像分类网络](https://www.jiqizhixin.com/articles/an-overview-of-image-classification-networks)
* [The 9 Deep Learning Papers You Need To Know About](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)






## 0. 数据处理

[github: DL-Data](https://github.com/jiye-ML/DL-Data)
* 当你训练一个机器学习模型时，你实际做工作的是调参，以便将特定的输入（一副图像）映像到输出（标签）。
我们优化的目标是使模型的损失最小化， 以正确的方式调节优化参数即可实现这一目标。



## 1. AlexNet

* [paper](paper/2012-AlexNet.pdf)

### 1.1 核心观点
* 在ImageNet数据集上训练网络, 15 million 标注图片 超过 22,000 类别.
* ReLU作为激活函数
* 数据增强技术: image translations, horizontal reflections, and patch extractions.
* dropout 防止 overfitting。
* 使用批随机梯度下降算法训练网络, 使用特殊值作为动量和权值衰减。
* Trained on two GTX 580 GPUs 5/6天。





## 2. VGG Net

* [paper](paper/2014-Very%20deep%20convolutional%20networks%20for%20large-scale%20image%20recognition.pdf)

* 核心观点
    * 仅仅使用3x3 卷积核，作者认为组合2个 3x3 卷积层和5x5核具有一样的感受野。这使得卷积核尺寸表现，感受野不变。
    一个好处是减少了参数量。
    * 随着输入空间尺寸的变小，通道数不断加大
    * 每次池化之后都是用多个卷积层，这诠释了空间尺度缩小的时候，深度加深。
    * 在训练过程中使用scale jittering作为一种数据增强技术。
    * 使用每个卷积层后使用ReLU层并且通过 batch gradient descent训练。

* 网络结构 \
![](readme/vgg_01.png)

* 比其他网络表现力更强，与gooleNet相比，网络架构更简单。
* 3*3网络的优势：
    1. 每一个卷积后面接ReLU，这样更加非线性
    2. 参数量少 3个3x3的卷积相当于一个7x7的卷积，感受野相同。
* 可以试着实现一些trick
    1. momentum SGD 0.9
    2. weight decay L2 * 5e-4
    3. drop out 0.5
    4. learning rate = 1e-2, 当验证集准确度不在提升时，衰减10， 衰减3次
    5. 初始化方式，预训练
* 初始化策略
    * 可以在一个小的网络上训练然后初始化
    * 训练11层的网络，然后利用11层的网络初始化相应的16层的网络
    * 可以利用256的数据训练，然后初始化384数据的网络




## 3. GooleNet

### Network in Network

* [paper](paper/2014-Network%20In%20Network.pdf)
    * 利用全局池化代替全连接
        * 目的是生成一张特征图为每个类别，
        * 每张特征图对应一个类别，更好的可解释性，
        * 没有需要优化的参数，避免了这一层的过拟合。
* [One by One [ 1 x 1 ] Convolution - counter-intuitively useful](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)
    * 1x1 的卷积网络具有减少深度的作用
    * 1x1卷积是线性的，但是后面一般都会接一个ReLU非线性激活单元。
    * 首先在 《Network in Network》论文中提出 \
    ![](readme/GoogLeNet_01.png)

    
### [v1] Going Deeper withConvolutions

* [paper](paper/2015-Going%20Deeper%20with%20Convolutions.pdf)
* 当我们观察GoogLeNet的结构的时候，我们注意到不是每件事都是序列发生的。\
![](readme/GoogLeNet_02.png)

* 核心观点
    * 第一个只是使用卷积和池化的堆叠的网络。
    * 使用9个Inception modules在整个网络中，超过100层.
    * 没有使用全连接层，使用全局平均池化将7x7x1024 volume到1x1x1024 volume. 节省了大量的参数。
    * 测试阶段, 使用multiple crops在一张图片上，softmax probabilities是平均值.


### [v2] Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate Shift

* [paper](paper/2015-Batch%20Normalization%20Accelerating%20Deep%20Network%20Training%20by%20Reducing%20Internal%20Covariate%20Shift.pdf)
* [利用随机前馈神经网络生成图像观察网络复杂度](https://blog.csdn.net/happynear/article/details/46583811)
    * 用实验的方法解释了 BN中 连个参数都很重要。
* [《Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift》论文解读](https://www.cnblogs.com/dmzhuo/p/5889157.html)
    1. 归一化：
    * 原因在于神经网络学习过程本质就是为了学习数据分布，一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；
    * 一旦每批训练数据的分布各不相同，那么网络就要在每次迭代都去学习适应不同的分布，这样将会大大降低网络的训练速度；
    2. 传递性
    * 网络的前面几层发生微小的改变，那么后面几层就会被累积放大下去。
    * 一旦网络某一层的输入数据的分布发生改变，那么这一层网络就需要去适应学习这个新的数据分布，
    所以如果训练过程中，训练数据的分布一直在发生变化，那么将会影响网络的训练速度。
    
    

### [v3] Rethinking theInception Architecture for Computer Vision

* [paper](paper/2015-Rethinking%20the%20Inception%20Architecture%20for%20Computer%20Vision.pdf)
* 论文做出的贡献主要有4个
    1. 分解大filters，使其小型化、多层化，其中有个“非对称卷积”很新颖
    2. 优化inception v1的auxiliary classifiers
    3. 提出一种缩小特征图大小的方法，说白了就是一种新的、更复杂的pooling层
    4. Label smooth，“标签平滑”，很难用中文说清楚的一种方法
* Szegedy还把一段时间内的科研心得总结了一下，在论文里写了4项网络设计基本原则：
    1. 尽量避免representational bottlenecks，这种情况一般发生在pooling层，字面意思是，pooling后特征图变小了，
    但有用信息不能丢，不能因为网络的漏斗形结构而产生表达瓶颈，解决办法是上面提到的贡献3
    2. 采用更高维的表示方法能够更容易的处理网络的局部信息，我承认前面那句话是我硬翻译的，principle 2我确实不太明白
    3. 把大的filters拆成几个小filters叠加，不会降低网络的识别能力，对应上面的贡献1
    4. 把握好网络深度和宽度的平衡，这个原则说了等于没说



### [v4] Inception-v4,Inception-ResNet and the Impact of Residual Connections on Learning

* [paper](paper/2016-Inception-v4,%20Inception-ResNet%20and%20the%20Impact%20of%20Residual%20Connections%20on%20Learning.pdf)
* 





## 4. ResNet

![](paper/resnet_01.jpg)


* [ImageNet: VGGNet, ResNet, Inception, and Xception with Keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
