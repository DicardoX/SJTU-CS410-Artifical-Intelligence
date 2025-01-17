# 卷积神经网络（`CNN`）相关知识

-----------------

## 卷积层输出大小尺寸计算

[参考链接：卷积层输出大小尺寸计算及 “SAME” 和 “VALID”](https://blog.csdn.net/weixin_37697191/article/details/89527315)

&emsp; 假设`input volume`为`M * N * k`，卷积层有`t`个过滤器，则`output volume`的尺寸一定为`f(M) * f(N) * t`，其中`f()`函数参见参考链接。

----------------

## 卷积层参数个数计算

&emsp; 计算公式为：**参数个数** = **卷积核尺寸**（二维）* **`input volume`的深度**（第三个参数，即上节中的`k`） *  **卷积核个数**

----------------------

## 池化层输出大小尺寸计算及参数个数计算

[参考链接：卷积神经网络经过卷积/池化之后的图像尺寸及参数个数以及FLOPs](https://blog.csdn.net/xiaohuihui1994/article/details/83375661)

-----------------------

## 简单的卷积神经网络 —— `LeNet`

[参考链接：LeNet详解](https://blog.csdn.net/qq_42570457/article/details/81460807)

------------------------

## 不同问题下激活函数的选择

[参考链接：常用激活函数比较](http://www.360doc.com/content/17/0927/09/10408243_690511166.shtml)

-----------------------
