title: 支持向量机笔记
date: 2017-02-20 22:14
categories: machine_learning
tags: [机器学习,支持向量机,SVM]
description: 对SVM学习的笔记，从硬间隔、软间隔、拉格朗日对偶法、核技巧、SMO算法等维度进行了记录。
---

SVM的笔记不好做，挣扎了好久还是记一记吧。

支持向量机（SVM，Support Vector Machine）是最好的有监督学习算法之一。一般需要讨论的点有：
- 硬间隔（hard margin），数据线性可分时通过硬间隔最大化可以学习到SVM分类器
- 软间隔（soft margin），数据近似线性可分时通过软间隔最大化可以学习到SVM分类器
- 拉格朗日对偶法（Lagrange duality），求解分类器参数时使用的方法
- 核技巧（kernels trick），数据线性不可分时，将特征映射到高维空间。
- 序列最小最优化（SMO）算法，一个快速求解SVM参数的算法

## 总体印象

训练数据集
$$\text{T} = \{( \boldsymbol{x}_1, y_1 ),(\boldsymbol{x}_2,y_2),\cdots,(\boldsymbol{x}_N,y_N)\}$$
其中，$\boldsymbol{x}_i \in R^n, \ y_i \in \{-1, +1\},\ i=1,2,\cdots,N$。$\boldsymbol{x}_i$ 为第 $i$ 个特征向量，也称为实例，$y_i$ 为 $\boldsymbol{x}_i$ 的类标记，当 $y_i = +1$ 时，称 $\boldsymbol{x}_i$ 为正例；当 $y_i = -1$ 时，称 $\boldsymbol{x}_i$ 为负例，$(\boldsymbol{x}_i, y_i)$ 称为样本点。

学习目标是在特征空间中找到一个分离超平面，能将实例分到不同的类。分离超平面对应于方程 $\boldsymbol{w}^{\text{T}} \boldsymbol{x} + b = 0$，它由法向量 $\boldsymbol{w}$ 和截距 $b$ 决定。分离超平面将特征空间划分为两部分，一部分是正类，一部分是负类。法向量指向的一侧为正类，另一侧为负类。

一般地，当训练数据集线性可分时，存在无穷个分离超平面可将两类数据正确分开。支持向量机利用间隔最大化求最优分离超平面，这时，解是唯一的。

## 硬间隔

先考虑数据集线性可分的二分类问题，即线性可分支持向量机。

### 函数间隔和几何间隔

**函数间隔** 对于给定的训练数据集 $\text{T}$ 和超平面 $\boldsymbol{w}^{\text{T}}\boldsymbol{x} + b = 0$，定义超平面关于样本点 $(x_i,y_i)$ 的函数间隔为
$$\hat{\gamma_i} = y_i(\boldsymbol{w}^{\text{T}}\boldsymbol{x}_i + b) \tag{1}$$
 定义超平面关于训练数据集 $\text{T}$ 的函数间隔为超平面关于 $\text{T}$ 中所有样本点 $(x_i, y_i)$ 的函数间隔之最小值，即
 $$\hat{\gamma} = \min_{i=1,\ldots,N} \hat{\gamma_i} \tag{2}$$

> 一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度。在超平面 $\boldsymbol{w}^{\text{T}} \boldsymbol{x} + b = 0$ 确定的情况下，$\mid \boldsymbol{w}^{\text{T}} \boldsymbol{x} + b \mid$ 能够相对地表示点 $\boldsymbol{x} $ 距离超平面的远近。而 $\boldsymbol{w}^{\text{T}} \boldsymbol{x} + b $ 的符号与类标记 $y$ 的符号是否一致能够表示分类是否正确。所以，可用量 $y(\boldsymbol{w}^{\text{T}}\boldsymbol{x} + b)$ 来表示分类的正确性及确信度，这就是函数间隔（functional margin）的概念。
> 但是，如果我们成比例的改变 $\boldsymbol{w}$ 和 $b$，例如将它们改为 $2\boldsymbol{w}$ 和 $2b$，超平面并没有改变，但函数间隔却成为原来的2倍。因此，要对$\boldsymbol{w}$加某些约束，如规范化，$\Vert \boldsymbol{w} \Vert = 1$，使得间隔是确定的。这时函数间隔成为几何间隔（geometric margin）。

**几何间隔** 对于给定的训练数据集 $\text{T}$ 和超平面 $\boldsymbol{w}^{\text{T}}\boldsymbol{x} + b = 0$，定义超平面关于样本点 $(x_i,y_i)$ 的几何间隔为
$$\hat{\gamma_i} = y_i \left(\frac{\boldsymbol{w}^{\text{T}}}{\Vert \boldsymbol{w} \Vert} \boldsymbol{x}_i + \frac{b}{\Vert \boldsymbol{w} \Vert} \right)  \tag{3}$$
 定义超平面关于训练数据集 $\text{T}$ 的几何间隔为超平面关于 $\text{T}$ 中所有样本点 $(x_i, y_i)$ 的几何间隔之最小值，即
 $$\gamma = \min_{i=1,\ldots,N} \gamma_i  \tag{4}$$

> 可以回顾高中三维空间中，点 $(x_0,y_0,z_0)$ 到平面 $Ax+By+Cz+D=0$ 的距离：
> $$ d = \frac{\mid Ax_0 + By_0 + Cz+0 +D \mid}{\sqrt{A^2 + B^2 + C^2}}$$
> 几何间距（式(3)）乘上 $y_i$ 的含义类似于上式中的绝对值，当样本被正确分类时，$y_i$ 与 $\frac{\boldsymbol{w}^{\text{T}}}{\Vert \boldsymbol{w} \Vert} \boldsymbol{x}_i + \frac{b}{\Vert \boldsymbol{w} \Vert}$ 的符号相同，相乘的结果肯定非负。
>
> 成比例修改 $\boldsymbol{w}$ 时，几何间距不会改变。

### 间隔最大化

求取几何间隔最大的超平面问题可以转化为如下的约束最优化问题：
$$ \begin{align*} \max_{\boldsymbol{w}, b} \quad &\gamma \tag{5} \\ \text{s.t.} \quad &y_i \left( \frac{\boldsymbol{w}^{\text{T}}}{\Vert \boldsymbol{w} \Vert} \boldsymbol{x}_i + \frac{b}{\Vert \boldsymbol{w} \Vert} \right) \geq \gamma,\  i=1,\cdots,N \tag{6} \end{align*}$$
即我们希望最大化超平面关于训练数据集的几何间距 $\gamma$，约束条件表示的是超平面关于每个训练样本点的几何间距至少是 $\gamma$。

根据几何间隔和函数间隔的关系，可将这个问题改写为
$$ \begin{align*} \max_{\boldsymbol{w}, b} \quad &\frac{\hat{\gamma}}{ \Vert \boldsymbol{w} \Vert } \tag{7} \\ \text{s.t.} \quad &y_i \left( \boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b \right) \geq \hat{\gamma},\  i=1,\cdots,N  \tag{8} \end{align*}$$
由于函数间隔的取值并不影响最优化问题的解（成比例改变 $\boldsymbol{w}$ 和 $b$ 时，函数间隔也会相应改变）。因此，可以取 $\hat{\gamma} = 1$，同时，注意到最大化 $\frac{1}{\Vert \boldsymbol{w} \Vert}$ 和最小化 $\Vert \boldsymbol{w} \Vert ^2$ 是等价的，最优化问题可以转化为：
$$ \begin{align*} \min_{\boldsymbol{w}, b} \quad &\frac{1}{2}\Vert \boldsymbol{w} \Vert^2 \tag{9} \\ \text{s.t.} \quad &y_i \left( \boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b \right) \geq 1,\  i=1,\cdots,N  \tag{10} \end{align*}$$
这是一个凸二次规划问题，可以直接求解。但通过拉格朗日对偶法可以更高效的求解，同时可以用上核技巧。

## 拉格朗日对偶法

对于式(9)、式(10)的含约束的最优化问题（原始最优化问题），每个不等式约束（式(10)）引进拉格朗日乘子（Lagrange multiplier）$\alpha_i \geq 0, \  i=1,2,\ldots,N$，构建拉格朗日函数（Lagrange function）：
$$L(\boldsymbol{w}, b, \alpha) = \frac{1}{2} \Vert \boldsymbol{w} \Vert^2 + \sum_{i=1}^{N} \alpha_i \left( 1 - y_i (\boldsymbol{w}^{\text{T}}\boldsymbol{x}_i + b) \right)  \tag{11} $$

### 原始问题与对偶问题

令
$$\theta_P(\boldsymbol{w}, b) = \max_{\alpha} L(\boldsymbol{w}, b, \alpha)  \tag{12} $$
则
$$\theta_P(\boldsymbol{w}, b) = \left\{ \begin{align*} \frac{1}{2} \Vert \boldsymbol{w} \Vert^2 &,\quad 约束条件满足 \\ \infty &, \quad 约束条件不满足 \end{align*} \right. $$
> 假设有一个约束条件不满足，即 $1 - y\_k (\boldsymbol{w}^{\text{T}}\boldsymbol{x}\_k + b) > 0$，我们可以令 $\alpha\_k = \infty$，则 $\theta\_P(\boldsymbol{w}, b) = \max\_{\alpha} L(\boldsymbol{w}, b, \alpha) = \infty$。
> 
> 如果约束条件全满足，则拉格朗日函数的第二项最大只可能为零，因此 $\theta\_P(\boldsymbol{w}, b) = \max\_{\alpha} L(\boldsymbol{w}, b, \alpha) = \frac{1}{2} \Vert \boldsymbol{w} \Vert^2$。

因此原始最优化问题等价于：
$$ \min_{\boldsymbol{w}, b} \theta_P (\boldsymbol{w}, b) = \min_{\boldsymbol{w}, b} \max_{\alpha} L(\boldsymbol{w}, b, \alpha) $$
设其解为 $p^\* = L(\boldsymbol{w}^\*, b^\*, \alpha^\*)$。

令
$$\theta_D(\alpha) = \min_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \alpha)  \tag{13} $$
得到对偶（dual）问题
$$ \max_{\alpha} \theta_D(\alpha) = \max_{\alpha} \min_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \alpha) $$
设其解为 $d^\* = L(\boldsymbol{w}^\*, b^\*, \alpha^\*)$。

$p^\*$ 和 $d^\*$ 存在如下关系：
$$d^* = \max_{\alpha} \min_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \alpha) \leq  \min_{\boldsymbol{w}, b} \max_{\alpha} L(\boldsymbol{w}, b, \alpha) = p^* \tag{14} $$

直接套用定理，通过以下条件可以推出存在 $\boldsymbol{w}^\*$，$b^\*$，$\boldsymbol{\alpha}^\*$，使 $\boldsymbol{w}^\*$，$b^\*$ 是原始问题的解，$\boldsymbol{\alpha}^\*$ 是对偶问题的解，并且 $p^\* = d^\*$（这里暂时还是糊涂的）：
- $\frac{1}{2}\Vert \boldsymbol{w} \Vert ^2$ 和 $1-y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b)$ 是凸函数
- 存在 $\boldsymbol{w}$ 和 $b$ 对所有的 $i$ 都有 $1-y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b) < 0$

$p^\* = d^\*$与下面的KKT条件是充要条件：
$$ \nabla_{\boldsymbol{w}} L(\boldsymbol{w}^*, b^*, \boldsymbol{\alpha}^*) = 0 \tag{15} $$
$$ \nabla_{b} L(\boldsymbol{w}^*, b^*, \boldsymbol{\alpha}^*) = 0 \tag{16} $$
$$ \alpha_i^* \left( y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b) - 1 \right) = 0, \quad i=1,2,\ldots,N \tag{17} $$
$$ 1 -  y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b) \leq 0 , \quad i=1,2,\ldots,N \tag{18}$$
$$ \alpha_i \geq 0, \quad i=1,2,\ldots,N \tag{19}$$

### 对偶问题求解

下面我们对对偶问题进行求解，之后我们就会明白为什么要费这么大劲转化为对偶问题。


1.  求 $\min_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \alpha)$
将拉格朗日函数（式(11)）别对 $\boldsymbol{w}$ 和 $b$ 求偏导并令其等于0。
$$ \nabla_{\boldsymbol{w}} L(\boldsymbol{w}, b, \alpha) =  \boldsymbol{w} - \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i = 0$$
$$ \nabla_{b} L(\boldsymbol{w}, b, \alpha) = \sum_{i=1}^{N}\alpha_i y_i = 0 $$
得
$$\boldsymbol{w} = \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i  \tag{20}$$
$$  \sum_{i=1}^{N}\alpha_i y_i = 0 \tag{21} $$
带入拉格朗日函数，得
$$ \begin{align*} L(\boldsymbol{w}, b, \alpha) 
&= \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^{\text{T}} \boldsymbol{x}_j - \sum_{j=1}^{N} \alpha_j y_j \left( \left( \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i \right)^{\text{T} } \boldsymbol{x}_j + b \right) + \sum_{i=1}^{N} \alpha_i \\
&= -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \left< \boldsymbol{x}_i, \boldsymbol{x}_j \right> + \sum_{i=1}^{N} \alpha_i
\end{align*}
$$
即
$$ \min_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \alpha) = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \left< \boldsymbol{x}_i, \boldsymbol{x}_j \right> + \sum_{i=1}^{N} \alpha_i$$
其中，$\left< \boldsymbol{x}_i, \boldsymbol{x}_j \right> = \boldsymbol{x}_i^{\text{T}} \boldsymbol{x}_j $表示两个向量的内积。

2. 求 $\min_{\boldsymbol{w},b} L(\boldsymbol{w},b,\alpha)$ 对 $\alpha$ 的极大：
$$ \begin{align*}
\max_{\alpha} \quad &-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \left< \boldsymbol{x}_i, \boldsymbol{x}_j \right> + \sum_{i=1}^{N} \alpha_i \\
\text{s.t.} \quad &\sum_{i=1}^{N}\alpha_i y_i = 0 \\
&\alpha_i \geq 0, \quad i=1,2,\ldots,N \end{align*} $$
可以等价于下面的求极小：
$$ \begin{align*}
\min_{\alpha} \quad &\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \left< \boldsymbol{x}_i, \boldsymbol{x}_j \right> - \sum_{i=1}^{N} \alpha_i \tag{22} \\
\text{s.t.} \quad &\sum_{i=1}^{N}\alpha_i y_i = 0 \tag{23} \\
&\alpha_i \geq 0, \quad i=1,2,\ldots,N \tag{24} \end{align*} $$
到这一步，可以按照优化方法求 $\boldsymbol{\alpha}$ 了。费这么大劲相对于式(9)、式(10)的直接求解来说貌似也没有什么明显的改进，但注意到上式中的**内积**，有了它我们就可以引入核技巧了，同时，这个最优化求解可以使用高效的SMO算法。

通过对偶法求解出 $\boldsymbol{\alpha}$ 后可以通过式(20)求出 $\boldsymbol{w}$。

由式(17)可以知道，$ y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b) - 1 \neq 0$ 时，$\alpha_i$ 必定等于0，这些数据在计算 $\boldsymbol{w}$ 时将不会再起作用。当 $\alpha_i$ 不等于0时，$ y_i(\boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b) - 1 = 0$ 必定成立，也就是起作用的点的函数间隔一定等于1，这些点就是支持向量。

随意选取一个支持向量，根据其函数间隔等于1，就可以算出参数 $b$，实际中可能会用所有支持向量取平均。

## Kernels

之前我们假设训练样本是线性可分的，即存在一个划分超平面能将训练样本正确分类。然而在现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本的超平面。例如下图中的“异或”问题就不是线性可分的。
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/svm/kernel_dimension.png)
对于这样的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分，例如将上面的异或问题映射到三维空间。

令 $\phi(\boldsymbol{x})$ 表示将 $\boldsymbol{x}$ 映射后的特征向量。特征空间中划分超平面所对应的的模型可表示为
$$\boldsymbol{w}^{\text{T}} \phi (\boldsymbol{x}) + b = 0$$
类似于式(9)、式(10)，有
$$ \begin{align*} \min_{\boldsymbol{w}, b} \quad &\frac{1}{2}\Vert \boldsymbol{w} \Vert^2 \\ \text{s.t.} \quad &y_i \left( \boldsymbol{w}^{\text{T}} \phi(\boldsymbol{x}_i) + b \right) \geq 1,\  i=1,\cdots,N  \end{align*}$$
其对偶问题
$$ \begin{align*}
\min_{\alpha} \quad &\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j  \phi(\boldsymbol{x}_i)^{\text{T}} \phi(\boldsymbol{x}_j)  - \sum_{i=1}^{N} \alpha_i \\
\text{s.t.} \quad &\sum_{i=1}^{N}\alpha_i y_i = 0 \\
&\alpha_i \geq 0, \quad i=1,2,\ldots,N \end{align*} $$

求解上式涉及到计算 $\phi(\boldsymbol{x}_i)^{\text{T}} \phi(\boldsymbol{x}_j)$，这是样本 $\boldsymbol{x}_i$ 与 $\boldsymbol{x}_j$ 映射到特征空间之后的内积。由于特征空间维数可能很高，甚至可能是无穷维，因此直接计算 $\phi(\boldsymbol{x}_i)^{\text{T}} \phi(\boldsymbol{x}_j)$ 通常是困难的。我们需要寻找一个函数满足下式来简化求解：
$$K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \left< \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \right> = \phi(\boldsymbol{x}_i)^{\text{T}} \phi(\boldsymbol{x}_j)$$

这里的 $K(\cdot, \cdot)$ 就是核函数（kernel function）。

也就是说，在核函数 $K(\boldsymbol{x}_i, \boldsymbol{x}_j)$ 给定的条件下，可以利用解线性分类问题的方法求解非线性分类问题的支持向量机。学习是隐式地在特征空间进行的，不需要显示地定义特征空间和映射函数。这样的技巧称为核技巧，它是巧妙地利用线性分类学习方法与核函数解决非线性问题的技术。在实际应用中，往往依赖领域知识直接选择核函数，核函数选择的有效性需要通过实验验证。

常见核函数：
- 线性核 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \boldsymbol{x}_i ^ {\text{T}} \boldsymbol{x}_j$
- 多项式核 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = (\boldsymbol{x}_i ^ {\text{T}} \boldsymbol{x}_j)^{d}$，其中 $d \geq 1$ 是多项式的次数
- 高斯核 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \exp \left( - \frac{\Vert \boldsymbol{x}_i - \boldsymbol{x}_j \Vert ^ {2}}{2\sigma^2} \right)$，其中 $\sigma > 0$ 为高斯核的带宽（width）
- 拉普拉斯核  $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \exp \left( - \frac{\Vert \boldsymbol{x}_i - \boldsymbol{x}_j \Vert}{\sigma} \right)$，其中 $\sigma > 0$
- Sigmoid核 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \tanh ( \beta \boldsymbol{x}_i ^ {\text{T}} \boldsymbol{x}_j + \theta)$，其中 $\tanh$ 为双曲正切函数，$\beta > 0, \theta<0$


## 软间隔

前面我们假设训练样本在样本空间或特征空间中是线性可分的，更普遍地，训练数据中有一些特异点（outlier），将这些特异点去除后，剩下大部分的样本组成的集合是线性可分的。因此，我们允许支持向量机在一些样本上出错，这就是软间隔（soft margin）的概念。

但对于出错的样本，我们要给它一定的损失，因此优化目标变为
$$ \min_{\boldsymbol{w},b} \frac{1}{2} \Vert \boldsymbol{w} \Vert ^2 + C \sum_{i=1}^{N} l_{0/1} \left( y_i \left( \boldsymbol{w}^T\boldsymbol{x}_i + b \right) - 1 \right) \tag{25}$$
其中 $C>0$ 是一个常数，$l_{0/1}$ 是“0/1损失函数”
$$l_{0/1}(z) = \left\{ \begin{align*} 1, &\quad \text{if} \ z<0; \\ 0, &\quad \text{otherwise.} \end{align*} \right.$$
当 $C$ 为无穷大时，式(25)迫使所有样本均满足约束(10)，此时式(25)等价式(9)；当 $C$ 取有限值时，式(25)允许一些样本不满足约束。

然而，$l\_{0/1}$ 非凸、非连续，数学性质不太好，使得式(25)不易直接求解。于是，人们通常用其他一些函数来代替 $l\_{0/1}$，称为“替代损失”（surrogate loss）。替代损失函数一般具有较好的数学性质，如它们通常是凸的连续函数且是 $l\_{0/1}$ 的上界。下面给出了三种常用的替代损失函数：
- hinge损失：$l_{hinge}(z) = \max(0, 1-z)$
- 指数损失（exponential loss）：$l_{exp}(z) = \exp (-z)$
- 对率损失（logistic loss）：$l_{log}(z) = \log (1+ \exp(-z))$

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/svm/svm_loss_function.png)












采用hinge（合页）损失，式(25)变成
$$ \min_{\boldsymbol{w},b} \quad \frac{1}{2} \Vert \boldsymbol{w} \Vert ^ 2 + C \sum_{i=1}^{N} \max \left( 0, 1 - y_i \left( \boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b \right) \right) \tag{26}$$
引入“松弛变量”（slack variable）$\xi_i \geq 0$，可将式(26)重写为
$$\begin{align*} 
\min_{\boldsymbol{w},b,\xi_i} \quad &\frac{1}{2} \Vert \boldsymbol{w} \Vert ^2 + C \sum_{i=1}^{N} \xi_i \tag{27}  \\
\text{s.t.} \quad &y_i\left( \boldsymbol{w}^{\text{T}} \boldsymbol{x}_i + b \right) \geq 1-\xi_i \\
&\xi_i \geq 0,\ i=1,2,\ldots,N
\end{align*}$$
这就是常用的“软间隔支持向量机”。

式(27)中每个样本都有一个对应的松弛变量，用以表征该样本不满足约束(10)的程度。但这仍然是一个二次规划问题。于是，类似于式(11)，通过拉格朗日乘子法可得到式(27)的拉格朗日函数
$$ \begin{align*}
L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})
= &\frac{1}{2} \Vert \boldsymbol{w} \Vert ^2 + C \sum_{i=1}^{N} \xi_i \\
&+ \sum_{i=1}^{N} \alpha_i \left( 1 - \xi_i - y_i \left( \boldsymbol{w}^{\text{T}} + b \right) \right) - \sum_{i=1}^{N} \mu_i \xi_i \tag{28} \end{align*} $$
其中 $\alpha_i \geq 0, \mu_i \geq 0$ 是拉格朗日乘子。

令 $L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})$ 对 $\boldsymbol{w}$、$b$、$\xi_i$ 的偏导为零可得
$$\boldsymbol{w} = \sum_{i=1}^{N} \alpha_i y_i \boldsymbol{x}_i $$
$$0 = \sum_{i=1}^{N} \alpha_i y_i$$
$$C = \alpha_i + \mu_i$$
带入式(28)得式(27)的对偶问题
$$\begin{align*}
\max_{\boldsymbol{\alpha}} \quad &\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^{\text{T}} \boldsymbol{x}_j \tag{29} \\
\text{s.t.} \quad &\sum_{i=1}^{N} \alpha_i y_i = 0 \tag{30} \\
&0 \leq \alpha_i \leq C, \ i=1,2,\ldots,N \tag{31}
\end{align*}$$
可以看到硬间隔的对偶问题（式(22)-式(24)）与软间隔的对偶问题（式(29)-式(31)）的唯一区别在于对偶变量的约束不同。

## SMO

上面我们推到了对偶问题的二次规划问题，它具有全局最优解，并且有许多最优化算法可用于求解这一问题，但是当训练样本容量很大时，这些算法往往变得非常低效，以致无法使用。所以，人们提出了一些快速算法，这里介绍序列最小最优化（Sequential Minimal Optimization）算法。