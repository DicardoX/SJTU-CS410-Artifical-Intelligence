# Project 1 : Drug Molecular Toxicity Prediction

&emsp; 使用`Kaggle`平台进行炼丹：[Kaggle使用教程](https://blog.csdn.net/qq_34851605/article/details/108238087)

&emsp; TensorFlow `Keras`官方教程：[Keras中文文档](https://keras.io/zh/) [Keras教程整理](https://www.jianshu.com/p/d02980fd7b54)

&emsp; `tf.keras.layers.conv2D()`函数解释：[函数教程](https://blog.csdn.net/godot06/article/details/105054657)

&emsp; TensorFlow `strides`参数详解：[strides参数详解](https://blog.csdn.net/TwT520Ly/article/details/79540251?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)

&emsp; 卷积神经网络（CNN）详解：[CNN详解](https://blog.csdn.net/tjlakewalker/article/details/83275322)

&emsp; `MaxPool2D()`：二维池化函数。通常来说，CNN的卷积层之间都会周期性地插入池化层。[CNN学习笔记：池化层](https://www.cnblogs.com/MrSaver/p/10356695.html)， [`MaxPool2D()`参数解释](https://blog.csdn.net/iblctw/article/details/107088462?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-5-107088462.nonecase&utm_term=maxpool2d%20参数)

&emsp; 激活函数`activation function`详解：[深度夜戏中几种常见的激活函数理解与总结](https://www.cnblogs.com/XDU-Lakers/p/10557496.html)

&emsp; 张量介绍：[【tensorflow】浅谈什么是张量`tensor`](https://blog.csdn.net/qq_31821675/article/details/79188449)

&emsp; `Onehot`独热编码详解：[机器学习：数据预处理之独热编码（One-Hot）](https://www.imooc.com/article/35900)
 
&emsp; TensorFlow `tf.reshape()`函数里的shape详解：[【Tensorflow】tf.reshape 函数](http://www.voidcn.com/article/p-quleiusw-bd.html)

&emsp; 全连接层详解：[CNN——全连接层](https://zhuanlan.zhihu.com/p/33841176)

&emsp; TensorFlow `tf.keras.layers.dense()`函数详解：[Tensorflow笔记之 全连接层tf.kera.layers.Dense()参数含义及用法详解](https://blog.csdn.net/Zh_1999a/article/details/107549858?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)

&emsp; TensorFlow `tf.losses.softmax_cross_entropy()`函数参数解释：[tf.losses.softmax_cross_entropy()及相邻函数中weights参数的设置](https://blog.csdn.net/weixin_42561002/article/details/87802096)

&emsp; TensorFlow `tf.train.AdamOptimizer().minimize(loss)`函数详解：可以自动进行学习率的衰减更新。 [tensorflow Optimizer.minimize()和gradient clipping](http://www.mamicode.com/info-detail-2375709.html)

&emsp; 训练时报错`InvalidArgumentError`解决：在训练开头加上`tf.reset_default_graph()`

&emsp; 理解卷积神经网络中的代（`epoch`）、迭代（`iteration`）和批大小（`batchsize`）的关系：[卷积神经网络训练三个概念（epoch，迭代次数，batchsize）](https://blog.csdn.net/qq_37274615/article/details/81147013)

&emsp; 模型训练过程中`batchsize`的选择：[模型训练中batch_size的选择](https://blog.csdn.net/tsq292978891/article/details/86720184)

&emsp; 模型训练过程中`learning rate`的变化选择：[【模型训练】如何选择最适合你的学习率变更策略](https://zhuanlan.zhihu.com/p/52608023)

&emsp; 模型训练过程中过拟合的解决方案：

 - 调整`Batchsize` [怎么选取训练神经网络时的Batch size?](https://www.zhihu.com/question/61607442)
 
 - 使用正则化缓解过拟合：[CNN学习笔记：正则化缓解过拟合](https://www.cnblogs.com/MrSaver/p/10217315.html)
 
 - TensorFlow中的损失函数选择：[Tensorflow 中的损失函数 —— loss 专题汇总](https://zhuanlan.zhihu.com/p/44216830)
 
 - `tf.reshape()函数`里`shape = [-1, 28, 28, 1]`的理解： [x = tf.reshape(x, shape=[-1, 28, 28, 1])的理解](https://blog.csdn.net/agent_snail/article/details/105700777?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf)
 
 &emsp; 最后一位对应最小的中括号，数字代表该中括号内有几个元素，其他类似。
 
```
z = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
                  
Input: Z = z.reshape([-1, 1, 1, 2]) 
Output: 
[[[[1 2]]]

 [[[3 4]]]

 [[[5 6]]]

 [[[7 8]]]]
 
Input: Z = z.reshape([-1, 2, 2, 2])
Output:
[[[[1 2]
   [3 4]]

  [[5 6]
   [7 8]]]]
   
Input: Z = z.reshape([-1, 2, 1, 1])
Output:
[[[[1]]

  [[2]]]

 [[[3]]

  [[4]]]

Input: Z = z.reshape([-1, 1, 2, 1])
Output:
[[[[1]
   [2]]]

 [[[3]
   [4]]]

 [[[5]
   [6]]]

 [[[7]
   [8]]]]

```

&emsp; `tf.keras`的卷积层正则化实现：[正则化实现方法tf&keras](https://blog.csdn.net/buziran/article/details/102726808)

&emsp; `tf.keras`的`dropout`实现：

```
self.dropout = tf.keras.layers.Dropout(0.01) # 随机丢弃层
```
 
 
