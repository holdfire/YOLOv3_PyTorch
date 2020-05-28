### YOLOv3_PyTorch
An implementation of YOLOv3 by PyTorch from [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/)  

### YOLOv3模型构造过程：  
（1）堆积卷积层，pooling层，shortcut层，rout层，Upsample层；  
（2）堆到了YOLO层，
（3）


### 3. YOLO的输出和损失函数
#### 3.1 YOLO的输出是什么？
&emsp;&emsp;YOLO的输出层是：
$$ Output: S \times S \times (B \times (xyhw + Confidence) + C)$$
其中，S表示输出网格的尺寸，YOLOV1中S=7表示原图被;B表示每个  
Confidence表示的是$Pr(Object) * IOU^{truth}_{pred}$的值，   
C代表的是各个类别的概率$Pr(Class_i|Object)$的值；
#### 3.2 输出对应的真实标记是什么？
&emsp;&emsp;
+ **对于confidence：**
&emsp;&emsp;首先每个bounding box需要打出label，如果有物体的中心落在了cell内，那么$Pr(object)=1$，因此confidence为$Pr(object)*IOU^{truth}_{pred}$；如果没有物体中心落在cell内，那么$Pr(object)=0$，$Pr(object) * IOU^{truth}_{pred}$也必然为0，因此confidence为0。注意这个IOU是在训练过程中不断计算出来的，网络在训练过程中预测的bounding box每次都不一样，所以和ground truth计算出来的IOU每次也会不一样。
+ **类别预测，类别是一个条件概率$Pr(class_i |object)$**
&emsp;&emsp;对于一个cell，如果物体的中心落在了这个cell，那么我们给它打上这个物体的类别label，并设置概率为1。换句话说，这个概率是存在一个条件的，这个条件就是cell存在物体。
+ **Bounding box预测**
&emsp;&emsp;bounding box的预测包括xywh四个值。xy表示bounding box的中心相对于cell左上角坐标偏移，宽高则是相对于整张图片的宽高进行归一化的。


#### 3.3 训练阶段：如何根据输出和真实标记计算损失？
&emsp;&emsp;**训练阶段的损失函数可以看成：两个detector和一个classifier共三个家伙的预测结果和真实值的区别，** 由以下部分组成： 
$$
\begin{aligned}
L = &\lambda_{\bold{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} l_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\
& + \sum_{i=0}^{S^2} \sum_{j=0}^{B} l_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{\bold{noobj}}  \sum_{i=0}^{S^2} \sum_{j=0}^{B} l_{ij}^{noobj} (C_i - \hat{C}_i)^2   \\
& + \sum_{i=0}^{S^2} l_{i}^{obj} \sum_{c \in classes}(p_i(c) - \hat p_i(c))^2
\end{aligned}
$$
其中，$l_{ij}^{obj}$表示第$i$个cell的第$j$个bounding box负责这个预测（针对训练阶段）；$l_{i}^{obj}$表示第$i$个cell包含有目标。   
&emsp;&emsp;大部分的预测框通常不包含目标，为了平衡损失函数，可以增大bounding box的损失，减小不包含目标的框的confidence损失，YOLO的损失函数设置了两个系数$\lambda_{coord}=5$以及$\lambda_{noobj}=0.5$来实现这种平衡。  
&emsp;&emsp;关于loss，需要特别注意的是需要计算loss的部分。并不是网络的输出都算loss，具体地说：
+ 有物体中心落入的cell，需要计算分类loss，两个predictor都要计算confidence loss，预测的bounding box与ground truth IOU比较大的那个predictor需要计算xywh loss？？？？？？？？？？？？？？？？？？？？？？？？？
+ 特别注意：没有物体中心落入的cell，只需要计算confidence loss。



#### 3.4 测试阶段
+ **对于confidence：**
&emsp;&emsp;在测试阶段，网络只是输出了confidece这个值，但它已经包含了$Pr(object)=0$，$Pr(object) * IOU^{truth}_{pred}$。因为你在训练阶段你给confidence打label的时候，给的是$Pr(object) * IOU^{truth}_{pred}$这个值，你在测试的时候，网络输出的也就是这个值。
+ **类别预测**
&emsp;&emsp;对于测试阶段来说，网络直接输出$Pr(class_i |object)$，就已经可以代表有物体存在的条件下类别概率。但是在测试阶段，作者还把这个概率乘上了confidence。也就是说我们预测的条件概率还要乘以confidence。为什么这么做呢？举个例子，对于某个cell来说，在预测阶段，即使这个cell不存在物体（即confidence的值为0），也存在一种可能：输出的条件概率 [公式]，但将confidence和 [公式] 乘起来就变成0了。这个是很合理的，因为你得确保cell中有物体（即confidence大），你算类别概率才有意义。
$$Pr(class_i |object) * Pr(object) * IOU^{truth}_{pred}$$


#### 3.5 损失函数的设计原理
**问题1： YOLOv1的一个cell只能预测一种类别的目标，为什么不设置为能够预测2个或多个类别？**    
&emsp;&emsp;如果一个cell要预测两类目标，那么这两个predictor要怎么分工，分别预测的是那两个目标？这个不知道啊，所以没办法这么做。而像faster rcnn这类算法，可以根据anchor与ground truth的IOU大小来安排anchor负责预测哪个物体，所以后来yolo2也采用了anchor思想，同个cell才能预测多个目标。
**问题2：既然一个cell只能预测一种类别的目标，那为什么又要用两个Bounding box来预测呢？**  
&emsp;&emsp;训练的时候会在线地计算每个predictor预测的bounding box和ground truth的IOU，计算出来的IOU大的那个predictor，就会负责预测这个物体，另外一个则不预测。这么做有什么好处？我的理解是，这样做的话，实际上有两个predictor来一起进行预测，然后网络会在线选择预测得好的那个predictor（也就是IOU大）来进行预测。通俗一点说，就是我找一堆人来并行地干一件事，然后我选干的最好的那个。

