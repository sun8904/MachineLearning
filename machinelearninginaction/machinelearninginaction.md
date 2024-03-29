# machinelearninginaction





**标称型**：标称型目标变量的结果只在有限目标集中取值，如真与假(标称型目标变量主要用于分类) 也就相当于枚举类型

**数值型**：数值型目标变量则可以从无限的数值集合中取值，如0.100，42.001等 (数值型目标变量主要用于回归分析)



# ch2 KNN

定义：根据最邻近的k个点的分类来决定分类结果

优点：对异常值不敏感，无数据输入假定，可以多分类

缺点： 计算复杂度高，空间复杂度高

适用范围：数值型，标称型

## 算法流程：

1.收集数据，准备数据，分析数据

2.测试算法  调整参数k

3.使用算法

没有训练步骤

## 分类核心过程：

1. 计算待测试点与数据集中点的距离
2. 排序选取距离最小的k个点
3. 选择k个点中出现频率最高的类别作为预测结果的分类

## 调整内容：

1. 参数K

2. 距离计算方式

   都会影响算法的结果，具体问题具体分析

[demo:](<https://github.com/sun8904/machinelearninginaction/tree/master/Ch02>) 



# ch3 决策树

定义 构造一颗可以分类的树

优点：计算复杂度不高，结果易理解，可以处理不相关的特征，对中间值缺失不敏感，可以多分类

缺点：过度匹配

## 算法流程：

1. 收集，准备，分析数据，数值型必须离散化

2. 训练：构造树
3. 测试: 测试集利用训练树来分类测试
4. 使用 不依赖训练集，只需要训练出的树结构。

## 核心：

信息增益：表示含信息量多少

每个特征按照信息量从大到小排序，按照特征来分类构造树，某个节点的准确率达到一定的阈值，就作为一个叶子节点。

## 调整内容：

满足一个叶子节点的条件，阈值越高，分类越准确，树也更复杂

对训练集也比较敏感



# ch4 朴素贝叶斯

利用先验概率（统计）来求后验概率（条件概率）的问题。

优点：数据少也有效（数据能反映整体情况下是个前提），可以多分类（可以算出每个类别的概率，当然可以支持多分类）

缺点：数据敏感（影响先验，就会影响后验，最终也会影响分类）

## 算法流程：

1. 收集，准备，分析
2. 训练：计算不同特征的条件概率
3. 测试：错误率
4. 使用：利用训练集计算的概率来验证，不依赖数据集，增量更新数据集就会影响概率了。

核心：

最常见的是文本分类，统计学来说，一个特征需要N个样本，

10个特征就是N的10次方，假设特征是独立的，就变成N*10个数据集了。

例子：留言文本分类，侮辱性，和非侮辱性
$$
P(C_i |w)=(P(w|C_i )P(C_i))/(P(w))
$$
ci：表示两个类别

w:对应的词向量

步骤：

1. 文本转成对应的词向量 （0,1）出现和未出现
2. 计算对应的概率，P(c) 每个类的概率，p(w|c)每个类下词的概率
3. 测试文本转成词向量，计算每个类别的概率得出分类结果 

## 调整内容：

p(w|c) 可能为零，分子分母各加一作为初始值。

出现改进到出现次数统计，就变成词袋模型，比0,1准确点

考虑正负样本比例



# ch5 逻辑回归 logistic 

本质上是线性回归用来分类的一种方法。回归就是拟合一组数据，进行预测，如果是线性方程，就是线性回归。

逻辑回归在后面加了一个步骤，对结果进行了一个sigmod函数映射到1，-1两个结果上，达到分类效果
$$
σ（z）=1/(1+e^(-z) )
$$

## 算法流程

1. 收集数据，准备数据，因为需要进行距离计算
2. 分析数据
3. 训练回归系数：梯度上升求解最大值
4. 测试算法： 分类比较快，不依赖训练集
5. 使用：数据，转换成计算用的，带入训练好的结果，得出结果。

## 核心：

梯度上升求解最大值，梯度下降求解最小值

为什么是求解最大值：按照上面的定义，在线性拟合中肯定是用梯度下降，让拟合的更准确。逻辑回归这边是要分类的最准确，可以看出分类结果最大化。 y‘是实际分类结果，f(x)是预测结果。分类的越准确y越大。
$$
y=y'*f(x)
$$
梯度上升算法：
$$
w≔w+α∇_w f(w)
$$
梯度下降：
$$
w≔w-α∇_w f(w)
$$
伪代码：

```
回归系数初始值1
重复R次：
​	 计算整个数据集梯度
​	 使用alpha*gradient更新回归系数
返回回归系数
```

迭代执行的停止条件：迭代次数，或者误差小于误差范围

## 调整内容：

随机梯度算法：一次只用一个样本点来更新系数，这样减少计算量

从迭代次数和系数的关系图上看，来回波动频繁，影响收敛速度。

改进随机梯度算法： 

1. 改变系数值不固定，alpha= 4/(1.0+i+j)+0.01， j迭代次数，i是样本点下标

2. 随机挑选样本点进行更新，并从数据集中移除。下次就不会选中了。

