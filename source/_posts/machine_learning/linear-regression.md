title: 机器学习之线性回归
date: 2016-11-10 17:52
categories: machine_learning
tags: 机器学习
description: 线性回归是机器学习中常用的回归方法，很多实际应用中会使用线性回归来构造基线版本。本文先对其概念进行了介绍，然后讨论了模型的求解方法，最后给出了线性回归在两个数据集上的Python实现。
---

## 概念

- 预测**连续值变量**
- 是**有监督学习**：样本是有输出值的。样本 $ D=\lbrace (x\_i,y\_i)\rbrace _{i=1}^{m} $
- 学习得到一个映射关系 $f:\mathcal{X}\rightarrow\mathcal{Y}$
- 假定输入x和输出y之间有线性相关关系


>**例**
假定房价只与面积有关
$$ f(x) = \theta_1 x+\theta_0 $$
>
假定房价只与面积和房间数量有关
$$ f(x_1,x_2) = \theta_0+\theta_1x_1+\theta_2x_2 $$

推广到n个变量（n个特征）：
$$\begin{align*}
f(\boldsymbol{x}) = h_{\boldsymbol{\theta}}(\boldsymbol{x}) &= \theta_0+\theta_1x_1+\cdots+\theta_nx_n \\
&= \left[ \theta_0, \theta_1,\theta_2,\cdots,\theta_n \right] \left[ \begin{matrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{matrix}\right] \\
&= \boldsymbol{\theta}^T\boldsymbol{x}
\end{align*}$$
其中 $ x_0 = 1 $，$\theta_0$ 称为截距，或者bias。

## 损失函数（loss function）

在线性回归的概念中已经知道了线性回归的模型，那么如何求解 $ \boldsymbol{\theta}$ 呢？关键在于如何衡量 $h\_\boldsymbol{\theta}(\boldsymbol{x})$ 与 $y$ 之间的差别。均方误差 $\sum \limits\_{i=1}^{m}(h\_\boldsymbol{\theta} (x\_{i})-y\_{i})^2$ 是回归任务中最常用的性能度量，因此我们可以试图让均方误差最小化。
$$ \boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\arg\min} \sum_{i=1}^{m}(h_\boldsymbol{\theta}(x_{i})-y_{i})^2$$
均方误差对应了常用的欧几里德距离（欧氏距离，Euclidean distance）。基于均方误差最小化进行模型求解的方法称为“最小二乘法”（least square method）。当向量$\boldsymbol{w}$为标量$w$时，最小二乘法就是试图找到一条直线，使所有样本到直线的欧式距离之和最小。

## 回归方程求解

### 求导（normal equation）

1. 只有一个特征
先讨论只有一个特征的情况。此时，
$$ h_\boldsymbol{\theta}(x) = \theta_1 x + \theta_0 $$
均方误差为：
$$ \begin{align*}
J(\theta_1, \theta_0) &= \sum_{i=1}^{m}(f(x_i) - y_i)^2 \\ 
&= \sum_{i=1}^{m}(\theta_1 x_i + \theta_0 - y_i)^2 
\end{align*}$$
将 $J(\theta_1, \theta_0)$ 分别对 $\theta_1$ 和 $\theta_0$ 求导，得
$$  \begin{align*}
\frac{\partial{J}}{\partial{\theta_1}} &= 2 \left( \theta_0\sum_{i=1}^{m}x_i^2 - \sum_{i=1}^{m} \left( y_i - \theta_0 \right) x_i \right) \\
\frac{\partial{J}}{\partial{\theta_0}} &= 2 \left( m\theta_0 - \sum_{i=1}^{m} \left( y_i - \theta_1x_i \right) \right) 
\end{align*}$$
令上式等于零可以得到 $\theta_1$ 和 $\theta_0$ 的闭式（closed-form）解：
$$ \begin{align*}
\theta_1 &= \frac{\sum \limits_{i=1}^{m}y_i \left( x_i - \overline{x} \right)}{\sum \limits_{i=1}^{m}x_i^2 - \frac{1}{m} \left( \sum \limits_{i=1}^{m}x_i \right)^2 } \\
\theta_0 &= \frac{1}{m}\sum_{i=1}^{m} \left( y_i - \theta_1x_i \right)
\end{align*}$$
其中 $\overline{x} = \frac{1}{m} \sum \limits_{i=1}^{m} x_i$ 是 $x$ 的均值。

2. 有多个特征（n个）
更一般地，有n个特征（$ n > 1 $）时，回归方程的形式为
$$ h_\boldsymbol{\theta}(x) = \boldsymbol{\theta}^T\boldsymbol{x} $$
均方误差表示成矩阵的形式：
$$ J(\boldsymbol{\theta}) = (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta})^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}) $$
其中
$$ \begin{align*}
\boldsymbol{X} = 
\left[ \begin{matrix}
 1 & \boldsymbol{x}_1^T \\ 1 & \boldsymbol{x}_2^T \\ \vdots & \vdots \\ 1 & \boldsymbol{x}_m^T \\
\end{matrix} \right]
=\left[  \begin{matrix} 1 & x_{11} & x_{12} & \cdots & x_{1n} \\ 1 & x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots & \vdots  \\ 1 & x_{m1} & x_{m2} & \cdots & x_{mn} \end{matrix} \right]
\end{align*}$$
对 $\boldsymbol{\theta}$ 求导得
$$\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 2\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\theta} - \boldsymbol{y})$$
令上式等于零，可以得到 $\boldsymbol{\theta}$ 最优的闭式解。当 $\boldsymbol{X}^T\boldsymbol{X}$ 为满秩矩阵或正定矩阵时，可解得
$$ \boldsymbol{\theta}^* =  (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

> 关于均方误差的矩阵形式求导：
> $$ \begin{align*} 
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
&= \frac{\partial \left( (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta})^T(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}) \right)}{\partial \boldsymbol{\theta}} \\
&= \frac{\partial \left( (\boldsymbol{y}^T - \boldsymbol{\theta}^T\boldsymbol{X}^T)(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}) \right)}{\partial \boldsymbol{\theta}} \\
&= \frac{\partial \left( \boldsymbol{y}^T\boldsymbol{y} - \boldsymbol{\theta}^T\boldsymbol{X}^T \boldsymbol{y} - \boldsymbol{y}^T\boldsymbol{X}\boldsymbol{\theta} + \boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta} \right)}{\partial \boldsymbol{\theta}} \\
&= \frac{\partial (\boldsymbol{y}^T\boldsymbol{y}) }{\partial \boldsymbol{\theta}} - \frac{\partial (\boldsymbol{\theta}^T\boldsymbol{X}^T \boldsymbol{y}) }{\partial \boldsymbol{\theta}} - \frac{\partial (\boldsymbol{y}^T\boldsymbol{X}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}} + \frac{\partial (\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}} \\
&=0 - \boldsymbol{X}^T \boldsymbol{y} - \boldsymbol{X}^T \boldsymbol{y} + 2\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta} \\
&= 2\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\theta} - \boldsymbol{y})
 \end{align*} $$
 > 其中，$\boldsymbol{\theta}^T\boldsymbol{X}^T \boldsymbol{y}$、$\boldsymbol{y}^T\boldsymbol{X}\boldsymbol{\theta}$、$\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}$ 都是标量（1×1的矩阵），所以可以分解成三个【标量对向量求导】的问题。我们知道对于含n特征的m个样本，$\boldsymbol{\theta}$、$\boldsymbol{X}$、$\boldsymbol{y}$  的维度分别是(n+1)×1、 m×(n+1)、m×1，为方便讨论，我们这里认为它们的维度分别为n×1、 m×n、m×1。三个标量分别对向量求导如下：
 > $$ \begin{align*}
\frac{\partial (\boldsymbol{\theta}^T\boldsymbol{X}^T \boldsymbol{y}) }{\partial \boldsymbol{\theta}}
\overset{\boldsymbol{X}^T \boldsymbol{y} = \boldsymbol{W}}{\Longrightarrow} \frac{\partial (\boldsymbol{\theta}^T\boldsymbol{W}) }{\partial \boldsymbol{\theta}}
&= \frac{\partial}{\partial \boldsymbol{\theta}} \left[ \begin{matrix} \theta_1 & \theta_2 & \cdots & \theta_n \end{matrix} \right] \left[ \begin{matrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{matrix} \right] \\
&= \frac{\partial}{\partial \boldsymbol{\theta}} \sum_{i=1}^{n}\theta_i w_i 
= \left[ \begin{matrix} \frac{\partial}{\partial \theta_1} \sum_{i=1}^{n}\theta_i w_i \\ \frac{\partial}{\partial \theta_2} \sum_{i=1}^{n}\theta_i w_i \\ \vdots \\ \frac{\partial}{\partial \theta_n} \sum_{i=1}^{n}\theta_i w_i \end{matrix} \right] 
= \left[ \begin{matrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{matrix} \right] = \boldsymbol{W} = \boldsymbol{X}^T \boldsymbol{y} \\
\\ \frac{\partial (\boldsymbol{y}^T\boldsymbol{X}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}}
\overset{\boldsymbol{y}^T\boldsymbol{X} = \boldsymbol{W}}{\Longrightarrow} \frac{\partial (\boldsymbol{W}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}}
&= \frac{\partial}{\partial \boldsymbol{\theta}}  \left[ \begin{matrix} w_1 & w_2 & \cdots & w_n \end{matrix} \right] \left[ \begin{matrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{matrix} \right] \\
&= \frac{\partial}{\partial \boldsymbol{\theta}} \sum_{i=1}^{n}w_i \theta_i
= \left[ \begin{matrix} \frac{\partial}{\partial \theta_1} \sum_{i=1}^{n}w_i \theta_i \\ \frac{\partial}{\partial \theta_2} \sum_{i=1}^{n}w_i \theta_i \\ \vdots \\ \frac{\partial}{\partial \theta_n} \sum_{i=1}^{n}w_i \theta_i \end{matrix} \right]
= \left[ \begin{matrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{matrix} \right] = \boldsymbol{W}^T = (\boldsymbol{y}^T\boldsymbol{X})^T =\boldsymbol{X}^T \boldsymbol{y} \\
\\ \frac{\partial (\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}}
\overset{\boldsymbol{X}^T\boldsymbol{X} = \boldsymbol{W}}{\Longrightarrow} \frac{\partial (\boldsymbol{\theta}^T\boldsymbol{W}\boldsymbol{\theta}) }{\partial \boldsymbol{\theta}}
&= \frac{\partial}{\partial \boldsymbol{\theta}} \left[ \begin{matrix} \theta_1 & \theta_2 & \cdots & \theta_n \end{matrix} \right] \left[ \begin{matrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nn} \end{matrix} \right] \left[ \begin{matrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{matrix} \right] \\
&= \frac{\partial}{\partial \boldsymbol{\theta}} \sum_{i=1}^{n} \sum_{j=1}^{n}\theta_i w_{ij} \theta_j
= \left[ \begin{matrix} \frac{\partial}{\partial \theta_1} \sum_{i=1}^{n} \sum_{j=1}^{n}\theta_i w_{ij} \theta_j \\ \frac{\partial}{\partial \theta_2} \sum_{i=1}^{n} \sum_{j=1}^{n}\theta_i w_{ij} \theta_j \\ \vdots \\ \frac{\partial}{\partial \theta_n} \sum_{i=1}^{n} \sum_{j=1}^{n}\theta_i w_{ij} \theta_j \end{matrix} \right] 
= \left[ \begin{matrix} 2\sum_{i=1}^{n}w_{1i}\theta_i \\ 2\sum_{i=1}^{n}w_{2i}\theta_i \\ \vdots \\ 2\sum_{i=1}^{n}w_{ni}\theta_i \end{matrix} \right] \\
&= 2 \left[ \begin{matrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nn} \end{matrix} \right] \left[ \begin{matrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{matrix} \right]
= 2\boldsymbol{W}\boldsymbol{\theta} = 2\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}
 \end{align*}$$
 > 其中，$\frac{\partial}{\partial \theta\_k} \sum \limits\_{i=1}^{n} \sum \limits\_{j=1}^{n}\theta\_i w\_{ij} \theta\_j$ 的求导需要详细叙述一下，注意到$w\_{ik} = w\_{ki}$，求导过程如下：
 > $$ \begin{align*}
\frac{\partial}{\partial \theta_k} \sum_{i=1}^{n} \sum_{j=1}^{n}\theta_i w_{ij} \theta_j
&= \frac{\partial}{\partial \theta_k} \left( \sum_{i=k}^{k} \sum_{j=k}^{k}\theta_i w_{ij} \theta_j + \sum_{i=k}^{k} \sum_{j=1,j\neq k}^{n}\theta_i w_{ij} \theta_j + \sum_{i=1,i\neq k}^{n} \sum_{j=k}^{k}\theta_i w_{ij} \theta_j + \sum_{i=1,i\neq k}^{n} \sum_{j=1,j\neq k}^{n}\theta_i w_{ij} \theta_j \right) \\
&= \frac{\partial}{\partial \theta_k} \left( w_{kk} \theta_k^2 + \sum_{j=1,j\neq k}^{n}\theta_k w_{kj} \theta_j + \sum_{i=1,i\neq k}^{n} \theta_i w_{ik} \theta_k + \sum_{i=1,i\neq k}^{n} \sum_{j=1,j\neq k}^{n}\theta_i w_{ij} \theta_j \right) \\
&= 2w_{kk}\theta_k +  \sum_{j=1,j\neq k}^{n} w_{kj} \theta_j + \sum_{i=1,i\neq k}^{n} \theta_i w_{ik} + 0 \\
&=  \sum_{i=1}^{n} w_{ki} \theta_i + \sum_{i=1}^{n} w_{ik}\theta_i \\
&= 2\sum_{i=1}^{n} w_{ki}\theta_i
\end{align*}$$

### 梯度下降（gradient descent）

很多时候直接通过求导求解闭式解并不容易，就需要使用梯度下降最优化方法来求最优解。

一般定义损失函数如下：
$$J(\boldsymbol{\theta})=\frac{1}{2m}\sum_{i=1}^{m}\left( h_{\boldsymbol{\theta}}(\boldsymbol{x}_{i})-y_{i} \right)^2$$

考虑损失函数的变化趋势。以只有一个特征且截距为零的情况 $ h_\theta(x) = \theta_1x $ 为例，下左图中红色叉叉代表三个样本，直线代表学习到的映射关系，可以发现，当 $\theta_1$ 逐渐增大时，损失函数的值先会逐渐减小，达到一个最小值后又会慢慢增大。
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/gradien_desent01.png)








有两个变量时，损失函数也有类似的变化趋势：
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/gradien_desent02.png)














利用梯度下降来逐步最小化损失函数，从而寻找估计参数的过程如下：

1. 先选定一组估计参数（即选定损失函数上某一点）
2. 求解出损失函数在这组参数下的梯度（即损失函数在该点变化最快的方向，只有一个变量时就是损失函数在该点的导数值（斜率），有两个变量时就是损失函数在该点对两个变量的偏导组成的方向向量）
3. 然后沿着梯度方向走一小步（学习率$\alpha$），得到新的参数值
4. 重复2、3，直到逼近损失函数的最小值（可以通过连续多次重复2、3后，损失函数的变化值都很小来判断逼近了最小值）

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/gradien_desent03.png)

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/gradien_desent04.png)

写成一个简单的伪代码的形式：
> repeat until convergence {
$\quad\quad \theta\_j:=\theta\_j-\alpha\frac{\partial}{\partial\theta\_j}J(\theta\_0,\theta\_1,\cdots,\theta\_n)$
>}
> 其中，
> $\theta\_0:=\theta\_0-\alpha\frac{1}{m}\sum\_{i=1}^{m}(h\_\boldsymbol{\theta}(\boldsymbol{x}\_i)-y\_i)$
> $\theta\_1:=\theta\_1-\alpha\frac{1}{m}\sum\_{i=1}^{m}(h\_\boldsymbol{\theta}(\boldsymbol{x}\_i)-y\_i)x\_{i1}$
> $\theta\_2:=\theta\_2-\alpha\frac{1}{m}\sum\_{i=1}^{m}(h\_\boldsymbol{\theta}(\boldsymbol{x}\_i)-y\_i)x\_{i2}$
> $\cdots$

这里偏导的计算比闭式解中偏导的计算要简单，详细推导就不展开了。

## 正则化（regularization）

### 欠拟合、过拟合


![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/lr-regularization.png)


**过拟合：**如果我们有特别多的特征，我们的假设函数曲线可以对原始数据拟合的非常好，但丧失了一般性，从而导致对新的待预测样本，预测效果差。如上右图。


### 正则化
防止过拟合的典型方法是正则化，在原损失函数的基础上加上一个正则化项（regularizer）或罚项（panalty term），下面是使用L2范数正则化项的损失函数：
$$J(\boldsymbol{\theta})=\frac{1}{2m}\left[\sum_{i=1}^{m} \left( h_\boldsymbol{\theta}(\boldsymbol{x}_i)-y_i \right)^2+\lambda\sum_{j=1}^{n}\theta_j^2\right]$$

## 实践

以斯坦福的Deep Learning公开课的习题为例，对线性回归进行了编码（Python）。

完整的代码以ipython notebook的形式放在了[github上](https://github.com/coder-ss/ml-learn/blob/master/linear-regression/lr-boys-height.ipynb)。


### 单特征问题

数据来自于斯坦福Deep Learning公开课的习题[Exercise: Linear Regression](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=DeepLearning&doc=exercises/ex2/ex2.html)，是一份2-8岁男孩身高的数据，用于寻找年龄与身高的关系。

读取数据、求解参数的代码如下。
``` python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 读取数据
x_arr = np.loadtxt('data/boys-height/x.dat')
x_arr.resize(len(x_arr), 1)
x_arr = np.append(np.ones((len(x_arr), 1)), x_arr, 1)
        
y_arr = np.loadtxt('data/boys-height/y.dat')
        
def gradient_descent(_x_arr, _y_arr, _theta_0, _theta_1, _alpha=0.07):
    """ 梯度下降函数
    
    :param _x_arr: 各个特征的值，包含了截距，所以在单特征时_x_arr的维度是 m×2
    :param _y_arr: 标签
    :param _theta_0: 
    :param _theta_1: 
    :param _alpha: 步长
    :return: 
    """
    _y_arr_predict = _x_arr[:,0] * _theta_0 + _x_arr[:,1] * _theta_1
    
    _theta_0 -= _alpha * np.mean(_y_arr_predict - _y_arr)
    _theta_1 -= _alpha * np.mean((_y_arr_predict - _y_arr) * _x_arr[:,1])
    
    return _theta_0, _theta_1

# 直接求解（normal equation）
x_mean = np.mean(x_arr[:,1])
theta_1 = np.sum(y_arr * (x_arr[:,1] - x_mean)) / (np.sum(x_arr[:, 1]**2) - np.sum(x_arr[:, 1])**2 / len(x_arr))
theta_0 = np.mean(y_arr - theta_1 * x_arr[:, 1])
print('通过normal equation求解得到的参数：theta_0=%s，theta_1=%s' % (theta_0, theta_1))

# 用梯度下降方法来求解
theta_0 = 0; theta_1 = 0  # 初始参数
theta_0_new = 0; theta_1_new = 0  # 迭代得到的参数
count = 1
while count < 2000:  # 迭代次数上限
    count += 1
    theta_0_new, theta_1_new = gradient_descent(x_arr, y_arr, theta_0, theta_1)
    # 如果迭代后 参数更新很小，就可以停止迭代了
    if abs(theta_0_new - theta_0) < 0.1e-7 and abs(theta_1_new - theta_1) < 0.1e-7:
        break
    theta_0 = theta_0_new
    theta_1 = theta_1_new
print('通过gradient descent迭代%s次后求解得到的参数：theta_0=%s，theta_1=%s' % (count, theta_0, theta_1))

# 绘制拟合直线
line_x = [2, 8]
line_y = [theta_0 + theta_1 * x for x in line_x]
plt.plot(line_x, line_y)
# 绘制样本
plt.scatter(x_arr[:, 1], y_arr)
plt.xlabel('Age in years'); plt.ylabel('Height in meters')

plt.show()
```
上述代码的输出如下：
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/lr-output01.png)

可以看到梯度下降算法和求导的方法的结果是一致的。

我们可以绘制出损失函数与参数的关系、损失函数的等高线，绘制代码和结果如下。

``` python
# 计算各个参数下的损失函数的值
theta_0_arr = np.arange(-3, 3, 0.06)
theta_1_arr = np.arange(-1, 1, 0.02)
z_vals = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        z_vals[i][j] = np.mean((theta_0_arr[j] + theta_1_arr[i] * x_arr[:, 1] - y_arr)**2) / 2

# 绘制损失函数与参数的关系
fig = plt.figure(1, figsize=(7, 7))
ax1 = fig.gca(projection='3d')
ax1.set_xlim([1, -1])
theta_1_arr, theta_0_arr = np.meshgrid(theta_1_arr, theta_0_arr)
ax1.plot_surface(theta_1_arr, theta_0_arr, z_vals,  rstride=2, cstride=1, cmap=cm.jet, linewidth=0.1, shade=True)
ax1.set_xlabel('$\\theta_1$', fontsize='x-large'); ax1.set_ylabel('$\\theta_0$', fontsize='x-large')
ax1.set_zlabel('Cost J', fontsize='x-large')

# 绘制损失函数的等高线
plt.figure(2, figsize=(6, 6))
CS = plt.contour(theta_1_arr, theta_0_arr,  z_vals, levels=np.logspace(-2, 2, 15))
plt.clabel(CS, inline=1, fontsize=8)
plt.xlabel('$\\theta_1$', fontsize='x-large'); plt.ylabel('$\\theta_0$', fontsize='x-large')

plt.show()
```
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/lr-output02.png)


























### 多特征问题

数据来自于斯坦福Deep Learning公开课的习题[Exercise: Multivariance Linear Regression](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=DeepLearning&doc=exercises/ex3/ex3.html)。该数据描述了房子面积、房间数量与房价的关系。

先读取和处理数据：
``` python
# 读取数据
x_arr = np.loadtxt('data/house-price/x.dat')
y_arr = np.loadtxt('data/house-price/y.dat')

# 处理数据
# 标准化处理
x_mean = np.mean(x_arr, 0)
x_std = np.std(x_arr, 0)
x_arr_scale = (x_arr - x_mean) / x_std
# 组装截距
x_arr_scale = np.append(np.ones((len(x_arr_scale), 1)), x_arr_scale, 1)
x_arr = np.append(np.ones((len(x_arr), 1)), x_arr, 1)
x_mat_scale = np.matrix(x_arr_scale)
x_mat = np.matrix(x_arr)
# 将标签表示成 m×1 的矩阵
y_mat = np.matrix(y_arr)
y_mat.resize((len(x_mat), 1))
```

然后寻找合适的学习率（alpha）。通过遍历几个不同的学习率，绘制学习率与均方误差的曲线图，可以发现学习率为1时收敛较快，而学习率过大（如2.1）时，均方误差不会收敛。
```
def mse(_x_mat, _y_mat, _thetas_mat):
    """ 求均方误差
    
    :param _x_mat: 各样本的特征值。m×(n+1) 的矩阵，这里例子是47×3
    :param _y_mat: 各样本的标签。m×1 的矩阵
    :param _thetas_mat: 模型参数。(n+1)×1 的矩阵。这里是3×1
    :return: 
    """
    _err = _x_mat * _thetas_mat - _y_mat
    rs = _err.T * _err / 2 / _y_mat.shape[0]
    
    return rs.flat[0]
    
def gradient_descent(_x_arr, _y_arr, _thetas, _alpha=2):
    """ 梯度下降函数
    
    :param _x_arr: 各样本的特征值。ndarray类型 
    :param _y_arr: 各样本的标签。ndarray类型 
    :param _thetas: 模型参数。ndarray类型 
    :param _alpha: 学习率
    :return: 更新后的模型参数
    """
    for j in range(_x_arr.shape[1]):
        err_list = [0.0] * _x_arr.shape[0]
        
        for i in range(_x_arr.shape[0]):
            err_list[i] = (np.sum(_x_arr[i, :] * _thetas) - _y_arr[i]) * _x_arr[i, j]

        _thetas[j] = _thetas[j] - _alpha * np.mean(err_list, 0)
    return _thetas

# 寻找学习率alpha
alphas = [0.01, 0.03, 0.1, 0.3, 1, 1.8]
plt.figure(1, figsize=(8, 6))
for alpha in alphas:
    count = 0
    mse_list = [0.0] * 200
    thetas = np.array([0.0, 0.0, 0.0])
    
    while count < 200:
        mse_list[count] = mse(x_mat_scale, y_mat, np.matrix(thetas).T)
        count += 1
        thetas = gradient_descent(x_arr_scale, y_arr, thetas.copy(), alpha)
    plt.plot(range(200), mse_list, label=alpha)
plt.legend()
plt.xlabel('Number of iterations', fontsize='medium'); plt.ylabel('Cost J', fontsize='medium')

# 学习率过大会发散
plt.figure(2, figsize=(8, 6))
count = 0
mse_list = [0.0] * 200
thetas = np.array([0.0, 0.0, 0.0])

while count < 200:
    mse_list[count] = mse(np.matrix(x_arr_scale), y_mat, np.matrix(thetas).T)
    count += 1
    thetas = gradient_descent(x_arr_scale, y_arr, thetas.copy(), 2.1)
plt.plot(range(200), mse_list, label=2.1)
plt.legend()
plt.xlabel('Number of iterations', fontsize='medium'); plt.ylabel('Cost J', fontsize='medium')
plt.show()
```
上述代码的输出：
![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/lr/lr-output03.png)




























最后用选好的学习率（alpha=1）和迭代次数（50）来求解。这里对比了梯度下降的方法和求导的方法，由于梯度下降时对特征作了标准化处理，得到的参数与求导的方法并不一致，但对于给定的测试用例，其计算结果是一样的。
```
# 选择学习步长 alpha=1，迭代 50 次
count = 0
thetas = np.array([0.0, 0.0, 0.0])
while count < 50:
    count += 1
    thetas = gradient_descent(x_arr_scale, y_arr, thetas.copy(), 1)
print('通过gradient descent得到的参数为：%s' % thetas, end='')

# 预测数据
x_predict = np.array([1650, 3])
x_predict_scale = (x_predict - x_mean) / x_std
x_predict_scale = np.append(np.ones(1), x_predict_scale, 0)
x_predict = np.append(np.ones(1), x_predict, 0)

# 
print('。预测值为：%s' % np.sum(thetas * x_predict_scale))

# 通过求导得到闭式解（normal equation）
thetas2 = np.array((x_mat.T*x_mat).I * x_mat.T * y_mat).ravel()
print('通过normal equation得到的参数为：%s' % thetas2, end='')
print('。预测值为：%s' % np.sum(thetas2 * x_predict))
```
上述代码的输出为：
`通过gradient descent得到的参数为：[ 340412.65957447  109447.79646964   -6578.35485416]。预测值为：293081.464335`
`通过normal equation得到的参数为：[ 89597.9095428     139.21067402  -8738.01911233]。预测值为：293081.464335`

## 参考资料

- julyedu机器学习班
- 机器学习 周志华
- 吴恩达老师的机器学习公开课
- 斯坦福Deep Learning公开课