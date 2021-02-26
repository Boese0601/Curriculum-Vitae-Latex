# 沉梦昂志的大冒险

## YoloV5

### .Train custom dataset

#### 1.Create dataset.yaml

download training initial document from:

```html
# download command/URL (optional)
download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco128/images/train2017/
val: ../coco128/images/train2017/

# number of classes
nc: 80

# class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']
```



#### 2.Create labels

Using Labelbox(COCO128 自带标注集)

Bitte achten Sie!

export your labels to **YOLO format**, 

with one `*.txt` file per image. The `*.txt` file specifications are:

1)One row per object

2)Each row is `class      x_center        y_center      width      height` format.

3)Box coordinates must be in **normalized xywh** format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.

4)Class numbers are zero-indexed (start from 0).

#### 3.Diretories

 `/coco128` is **next to** the `/yolov5` directory.

For instance:

```html
coco/images/train2017/000000109622.jpg  # image
coco/labels/train2017/000000109622.txt  # label
```

#### 4.Model

pretrained model

download from:https://github.com/ultralytics/yolov5/releases

Select yolov5s(smallest and fastest)

#### 5.Train.py

Config the model in train.py

```python
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
parser.add_argument('--project', default='runs/train', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
```

#### 6.View result 

PATH:/runs/train/exp5

weights:Trained Model: best.pt

#### 7.detect.py

Using detect.py to use the trained model in real time detecting(Webcam/Video/Picture)

config detect.py

```python
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
```

### .Model Ensembling







## Object Detection--YoloV3

### 1.Yolo v1 算法

将输入图像分成S×S的网格。如果一个目标的中心落入一个网格单元中，该网格单元负责检测该目标。

每个格子预测B个bounding box及其置信度(confidence score)，以及C个类别概率。（一个box包含五个信息）

最后输出（SxSx(B*5+C))的Tensor

![yolo-grid-predict](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181242586-322645739.png)

**<u>置信度：</u>**

反映是否包含物体以及包含物体情况下位置的准确性,定义为
$$
Pr(Object) \times IOU^{truth}_{pred}, 其中Pr(Object)\in\{0,1\}
$$


#### 1.网络结构

借鉴了GoogLeNet分类网络结构。

不同的是，YOLO未使用inception module，而是使用1x1卷积层（此处1x1卷积层的存在是为了跨通道信息整合）+3x3卷积层简单替代。

YOLOv1网络比VGG16快(浮点数少于VGG的1/3),准确率稍差。

缺点：

输入尺寸固定：由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。其它分辨率需要缩放成改分辨率

占比较小的目标检测效果不好.虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。



#### 2.Loss Function

$$
\text{loss=$\sum_{i=0}^{s^2}$coordErr+iouErr+clsErr}
$$

简单相加时还要考虑每种loss的贡献率,YOLO给coordErr设置权重λcoord=5λcoord=5.在计算IOU误差时，包含物体的格子与不包含物体的格子，二者的IOU误差对网络loss的贡献值是不同的。若采用相同的权值，那么不包含物体的格子的confidence值近似为0，变相放大了包含物体的格子的confidence误差在计算网络参数梯度时的影响。为解决这个问题，YOLO 使用λnoobj=0.5修正iouErr。（此处的‘包含’是指存在一个物体，它的中心坐标落入到格子内）。对于相等的误差值，大物体误差对检测的影响应小于小物体误差对检测的影响。这是因为，相同的位置偏差占大物体的比例远小于同等偏差占小物体的比例。YOLO将物体大小的信息项（w和h）进行求平方根来改进这个问题，但并不能完全解决这个问题。

综上，YOLO在训练过程中Loss计算如下式所示：

![yolo-loss](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181317188-1434000633.png)

其中有宝盖帽子符号
$$
\hat x,\hat y,\hat w,\hat h,\hat C,\hat p
$$
为预测值,无帽子的为训练标记值。
$$
\mathbb 1_{ij}^{obj}
$$
表示物体落入格子i的第j个bbox内.如果某个单元格中没有目标,则不对分类误差进行反向传播;B个bbox中与GT具有最高IoU的一个进行坐标误差的反向传播,其余不进行.

#### 3.Training Process

1）预训练。使用 ImageNet 1000 类数据训练YOLO网络的前20个卷积层+1个average池化层+1个全连接层。训练图像分辨率resize到224x224。

2）用步骤1）得到的前20个卷积层网络参数来初始化YOLO模型前20个卷积层的网络参数，然后用 VOC 20 类标注数据进行YOLO模型训练。检测通常需要有细密纹理的视觉信息,所以为提高图像精度，在训练检测模型时，将输入图像分辨率从224 × 224 resize到448x448。

训练时B个bbox的ground truth设置成相同的.

#### 4.与Faster—RCNN

YOLO与Fast R-CNN相比有较大的定位误差，与基于region proposal的方法相比具有较低的召回率。但是，YOLO在定位识别背景时准确率更高，而 Fast-R-CNN 的假阳性很高。基于此作者设计了 Fast-R-CNN + YOLO 检测识别模式，即先用R-CNN提取得到一组bounding box，然后用YOLO处理图像也得到一组bounding box。对比这两组bounding box是否基本一致，如果一致就用YOLO计算得到的概率对目标分类，最终的bouding box的区域选取二者的相交区域。这种组合方式将准确率提高了3个百分点。

### 2.Yolo v2

#### 1.改进

1. **Batch Normalization**： v1中也大量用了Batch Normalization，同时在定位层后边用了dropout，v2中取消了dropout，在卷积层全部使用Batch Normalization。

2. **高分辨率分类器**：v1中使用224 × 224训练分类器网络，扩大到448用于检测网络。v2将ImageNet以448×448 的分辨率微调最初的分类网络，迭代10 epochs。

3. **Anchor Boxes**：v1中直接在卷积层之后使用全连接层预测bbox的坐标。v2借鉴Faster R-CNN的思想预测bbox的偏移.移除了全连接层,并且删掉了一个pooling层使特征的分辨率更大一些.另外调整了网络的输入(448->416)以使得位置坐标是奇数只有一个中心点(yolo使用pooling来下采样,有5个size=2,stride=2的max pooling,而卷积层没有降低大小,因此最后的特征是416/(2^5)=13).v1中每张图片预测7x7x2=98个box,而v2加上Anchor Boxes能预测超过1000个.检测结果从69.5mAP,81% recall变为69.2 mAP,88% recall.

   YOLO v2对Faster R-CNN的手选先验框方法做了改进,**采样k-means在训练集bbox上进行聚类产生合适的先验框**.由于使用欧氏距离会使较大的bbox比小的bbox产生更大的误差，而IOU与bbox尺寸无关,因此使用IOU参与距离计算,使得通过这些anchor boxes获得好的IOU分值。距离公式：
   $$
   D\text{(box,centroid)} = 1 − IOU\text{(box,centroid)}
   $$
   使用聚类进行选择的优势是达到相同的IOU结果时所需的anchor box数量更少,使得模型的表示能力更强,任务更容易学习.k-means算法代码实现参考:[k_means_yolo.py](https://github.com/PaulChongPeng/darknet/blob/master/tools/k_means_yolo.py).算法过程是:将每个bbox的宽和高相对整张图片的比例(wr,hr)进行聚类,得到k个anchor box,由于darknet代码需要配置文件中region层的anchors参数是绝对值大小,因此需要将这个比例值乘上卷积层的输出特征的大小.如输入是416x416,那么最后卷积层的特征是13x13.

4. **细粒度特征**(fine grain features):在Faster R-CNN 和 SSD 均使用了不同的feature map以适应不同尺度大小的目标.YOLOv2使用了一种不同的方法，简单添加一个 pass through layer，把浅层特征图（26x26）连接到深层特征图(连接到新加入的三个卷积核尺寸为3 * 3的卷积层最后一层的输入)。 通过叠加浅层特征图相邻特征到不同通道（而非空间位置），类似于Resnet中的identity mapping。这个方法把26x26x512的特征图叠加成13x13x2048的特征图，与原生的深层特征图相连接，使模型有了细粒度特征。此方法使得模型的性能获得了1%的提升。

5. **Multi-Scale Training**: 和YOLOv1训练时网络输入的图像尺寸固定不变不同，YOLOv2（在cfg文件中random=1时）每隔几次迭代后就会微调网络的输入尺寸。训练时每迭代10次，就会随机选择新的输入图像尺寸。因为YOLOv2的网络使用的downsamples倍率为32，所以使用32的倍数调整输入图像尺寸{320,352，…，608}。训练使用的最小的图像尺寸为320 x 320，最大的图像尺寸为608 x 608。 这使得网络可以适应多种不同尺度的输入.


#### 2.网络结构

**分类网络**

YOLOv2提出了一种新的分类模型Darknet-19.借鉴了很多其它网络的设计概念.主要使用3x3卷积并在pooling之后channel数加倍(VGG);global average pooling替代全连接做预测分类,并在3x3卷积之间使用1x1卷积压缩特征表示(Network in Network);使用 batch normalization 来提高稳定性,加速收敛,对模型正则化.
Darknet-19的结构如下表:

![Darknet-19-arch](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181344634-594145493.png)

**检测网络**

在分类网络中移除最后一个1x1的层,在最后添加3个3x3x1024的卷积层,再接上输出是类别个数的1x1卷积.
对于输入图像尺寸为`Si x Si`,最终3x3卷积层输出的feature map是`Oi x Oi`(Oi=Si/(2^5)),对应输入图像的Oi x Oi个栅格，每个栅格预测`#anchors`种boxes大小，每个box包含4个坐标值,1个置信度和`#classes`个条件类别概率，所以输出维度是`Oi x Oi x #anchors x (5 + #classes)`。

添加**跨层跳跃连接**（借鉴ResNet等思想）,融合粗细粒度的特征:将前面最后一个3x3x512卷积的特征图,对于416x416的输入,该层输出26x26x512,直接连接到最后新加的三个3x3卷积层的最后一个的前边.将26x26x512变形为13x13x1024与后边的13x13x1024特征按channel堆起来得到13x13x3072.从yolo-voc.cfg文件可以看到，第25层为route层，逆向9层拿到第16层26 * 26 * 512的输出，并由第26层的reorg层把26 * 26 * 512 变形为13 * 13 * 2048，再有第27层的route层连接24层和26层的输出，堆叠为13 * 13 * 3072，由最后一个卷积核为3 * 3的卷积层进行跨通道的信息融合并把通道降维为1024。

```python
learning_rate=0.0001
batch=64 
max_batches = 45000 # 最大迭代batch数 
policy=steps # 学习率衰减策略
steps=100,25000,35000 # 训练到这些batch次数时learning_rate按scale缩放 
scales=10,.1,.1 # 与steps对应

```

网络结构如下(输入416,5个类别,5个anchor box; 此结构信息由Darknet框架启动时输出):![YOLO v2-network](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181407843-936091130.png)

### 3.Yolo v3

#### 1.改进

**分类器-类别预测**：
YOLOv3不使用Softmax对每个框进行分类，主要考虑因素有两个：

1. Softmax使得每个框分配一个类别（score最大的一个），而对于`Open Images`这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类。
2. Softmax可被独立的多个logistic分类器替代，且准确率不会下降。
   分类损失采用binary cross-entropy loss.

**多尺度预测**
每种尺度预测3个box, anchor的设计方式仍然使用聚类,得到9个聚类中心,将其按照大小均分给3中尺度.

- 尺度1: 在基础网络之后添加一些卷积层再输出box信息.
- 尺度2: 从尺度1中的倒数第二层的卷积层上采样(x2)再与最后一个16x16大小的特征图相加,再次通过多个卷积后输出box信息.相比尺度1变大两倍.
- 尺度3: 与尺度2类似,使用了32x32大小的特征图.

参见网络结构定义文件[yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

**基础网络 Darknet-53**
仿ResNet, 与ResNet-101或ResNet-152准确率接近,但速度更快.对比如下:

![darknet-53 compare](https://images2018.cnblogs.com/blog/606386/201803/606386-20180327003709767-1829778920.png)

Darknet-53：![YOLOv3-arch](https://images2018.cnblogs.com/blog/606386/201803/606386-20180327004340505-1572852891.png)

YOLOv3在 mAP0.5mAP0.5 及小目标 APSAPS 上具有不错的结果,但随着IOU的增大,性能下降,说明YOLOv3不能很好地与ground truth切合.（Yolo通病）

#### 2.优缺点

优点

- 快速,pipline简单.
- 背景误检率低。
- 通用性强。YOLO对于艺术类作品中的物体检测同样适用。它对非自然图像物体的检测率远远高于DPM和RCNN系列检测方法。

但相比RCNN系列物体检测方法，YOLO具有以下缺点：

- 识别物体位置精准性差。
- 召回率低。在每个网格中预测固定数量的bbox这种约束方式减少了候选框的数量。

### **4.YOLO VS Faster-RCNN**

统一网络:
YOLO没有显示求取region proposal的过程。Faster R-CNN中尽管RPN与fast rcnn共享卷积层，但是在模型训练过程中，需要反复训练RPN网络和fast rcnn网络.
相对于R-CNN系列的"看两眼"(候选框提取与分类，图示如下),YOLO只需要Look Once.

YOLO统一为一个回归问题
而R-CNN将检测结果分为两部分求解：物体类别（分类问题），物体位置即bounding box（回归问题）。

![R-CNN pipline](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181429804-1383715883.png)

### 

## Pysot Tracking



## 张士峰物体检测

### Chapter 1.物体检测概述

分类--分类+定位--物体检测--实例分割

Object Detection

![image.238EW0](沉梦昂志的大冒险.assets/image.238EW0-1608859351069.png)

#### Developing

![image.DTTDW0](沉梦昂志的大冒险.assets/image.DTTDW0-1608859373522.png)

![2020-12-25 09-24-47屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-24-47屏幕截图.png)

Traditional Methods![2020-12-25 09-27-26屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-27-26屏幕截图.png)

Faster RCNN(With Bounding Box)

![2020-12-25 09-28-45屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-28-45屏幕截图.png)

CornerNet & Anchor-Free

![2020-12-25 09-33-45屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-33-45屏幕截图.png)

#### Evaluation Method (Precision)

##### **Recall Rate & mAP**

Recall = Recall True Positive / Positive Total

Precision = True Positive/ Detection Result![2020-12-25 09-41-44屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-41-44屏幕截图.png)

mAP = Precision-Recall 面积![2020-12-25 09-44-55屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-44-55屏幕截图.png)

mmAP ：IoU >= [0.5 0.55 0.6 … 0.9 ]的mAP再求平均值![2020-12-25 09-47-34屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-47-34屏幕截图.png)

##### MR^-2 

 (漏检率-虚检个数的面积)![2020-12-25 09-52-50屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 09-52-50屏幕截图.png)

#### Evaluation Method (Speed)

##### ms

前传耗时 (ms):从输入一张图像到输出最终结果所消耗的时间,这包括前处理耗时 (如图像归一化)、网络前传耗时、后处理耗时(如非极大值抑制)

##### FPS

每秒帧数 (FPS):每秒钟能处理的图像数量

##### FLOPs

浮点运算量 (FLOPs):处理一张图像所需要的浮点运算数量,它跟软硬件没有任何关系,可以公平地比较不同算法之间的检测速度

### Chapter 2.通用物体检测

• 物体检测:找出一副图像上感兴趣的物体,并给出它们的类别和位置
• 通用物体检测:感兴趣物体是很多类的物体
• 特定物体检测:感兴趣物体是某一类或某一大类物体![2020-12-25 10-53-09屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 10-53-09屏幕截图.png)

### Chapter3.Anchor  Algorithm

1.框设计

2.框分类

3.框回归

4.框预测

5.框匹配

![2020-12-25 10-59-37屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 10-59-37屏幕截图.png)

#### R-CNN

##### R-CNN的检测步骤:

1 输入图像:输入一张待检测的图像
2 候选区域生成:使用Selective Search算法,在输入图像上生成~2K个候选区域
3 候选区域处理:在输入图像上裁剪出每个候选区域,并缩放到227*227大小
4 特征提取:每个候选区域输入CNN网络提取一定维度(如4096维)的特征
5 类别判断:把提取的特征送入每一类的SVM 分类器,判别是否属于该类
6 位置精修:使用回归器精细修正候选框位置![image.GD1AW0](沉梦昂志的大冒险.assets/image.LLJ6V0.png)

##### 相比传统方法,检测精度得到大幅度提升,但是速度太慢,原因是:

1 使用Selective Search生成候选区域非常耗时
2 一张图像上有~2K个候选区域,需要使用~2K次CNN来提取特征,存在大量重复计算
3 特征提取、图像分类、边框回归是三个独立的步骤,要分别训练,测试效率也较低

#### Fast RCNN

##### Fast R-CNN的检测步骤:

1 输入图像:输入一张待检测的图像
2 候选区域生成:使用Selective Search算法,在输入图像上生成~2K个候选区域
3 特征提取:将整张图像传入CNN提取特征
4 候选区域特征:利用RoIPooling分别生成每个候选区域的特征
5 候选区域分类和回归:利用扣取的特征,对每个候选区域进行分类和回归

* 注:步骤45仍会存在一些重复计算,但是相对R-CNN少了很多![image.MLJ5V0](沉梦昂志的大冒险.assets/image.MLJ5V0.png)

为什么要使用RoIPooling把不同大小的特征变成固定大小?
1 网络后面是全连接层( FC层),要求输入有固定的维度
2 各个候选区域的特征大小一致,可以组成batch进行处理

##### 特点：

◼端到端的多任务训练
◼ Fast R-CNN比R-CNN快了200多倍,并且精度更高
◼ 生成候选区域算法( Selective Search)非常慢(耗时2s )
So ：采用Faster R-CNN

#### Faster R-CNN

◼ Fast R-CNN= *Selective Search* + Fast R-CNN
◼ Faster R-CNN = <u>**RPN**</u> +Fast R-CNN
◼ RPN取代了耗时的Selective Search
◼ RPN与Fast RCNN共享卷积层
◼ 引入计算量小,耗时少
◼ 可以端到端地训练
◼ Faster R-CNN确定了基于锚框算法的检测流程

#### ![image.81X9V0](沉梦昂志的大冒险.assets/image.81X9V0.png)

##### RPN流程：

![2020-12-25 11-16-15屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-16-15屏幕截图-1608866320991.png)

![2020-12-25 11-16-15屏幕截图](图片/2020-12-25 11-16-42屏幕截图.png)

![2020-12-25 11-16-49屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-19-12屏幕截图.png)



![2020-12-25 11-19-01屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-19-01屏幕截图-1608866445275.png)



![2020-12-25 11-19-12屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-19-12屏幕截图-1608866452262.png)

0

![2020-12-25 11-21-27屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-21-27屏幕截图.png)



![2020-12-25 11-25-27屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-25-27屏幕截图.png)



![2020-12-25 11-35-42屏幕截图](沉梦昂志的大冒险.assets/2020-12-25 11-35-42屏幕截图.png)
