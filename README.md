# SJTU-CS410-Artifical-Intelligence

 - Project 1: `Drug Molecular Toxicity Prediction`
 
 --------------------
 
 ## 过拟合 vs 欠拟合

[机器学习中的偏差(bias)和方差(variance)](https://blog.csdn.net/mingtian715/article/details/53789487)

 - 欠拟合：偏差`bias`大。模型修改策略：增多数据特征数、添加高次多项式特征、减小正则化系数λ
 
 - 过拟合：方差`variance`大。模型修改策略：增大数据规模、减小数据特征数（维数）、增大正则化系数λ

------------------------

## 受试者工作特征曲线`ROC`和曲线下面积`AUC`

[如何理解机器学习和统计中的AUC？](https://www.zhihu.com/question/39840928)

 - 给定预测标签下的`ROC`曲线绘制及`AUC`计算
 
 - 给定概率型预测标签下的`ROC`曲线绘制及`AUC`计算：**按照预测值相对大小顺序依次作为阈值**，分别计算多组出`TPR`和`FPR`之后，在`[0, 1] * [0, 1]`坐标系中以`FPR`作为横坐标，`TPR`作为纵坐标标点，再将所有点依次相连。注意一定会包含`(0, 0)`和`(1, 1)`两个点。
