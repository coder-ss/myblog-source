title: 机器学习之矩阵基础
date: 2016-05-10 21:35
categories: machine_learning
tags: [机器学习,线性代数,矩阵分析]
description: 线性代数、矩阵论中的一些知识。
---

## Ax=b 的两种理解

> 基于列视图来理解 $A\boldsymbol{x}=b$，可以更好的理解矩阵。

分别对于二元线性方程组
$$ \left\{
\begin{align*}
2x-y &=1 \\
x+y &= 5 \\
\end{align*}
\right.
$$

和三元线性方程组
$$ \left\{
\begin{align*}
2x+y+z &=5 \\
4x-6y &= -2 \\
-2x+7y+2z&=9\\
\end{align*}
\right.
$$

分别讨论他们的**行视图**与**列视图**。

### 行视图

二元线性方程组理解为：在 $ XOY $ 坐标轴系中直线 $ 2x-y=1 $ 与 直线 $ x+y=5 $ 的交点。

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/2line.png)









三元线性方程组理解为：在 $ XYZ $ 坐标系中平面 $ 2x+y+z =5 $ 与平面 $ 4x-6y = 0 $ 与平面 $ -2x+7y+2z=9 $  的交点（前两个平面交于一条直线，这条直线与第三个平面交于一点）。

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/3plant.png)










### 列视图

$$ 	x\left[ \begin{aligned}2 \\ 1 \\ \end{aligned}\right]+y\left[ \begin{aligned}-1 \\ 1 \\ \end{aligned}\right]=\left[ \begin{aligned}1 \\ 5 \\ \end{aligned}\right]$$
 
 
二元线性方程组理解为：向量 $ \left[ \begin{aligned}2 \\\\ 1 \\\\ \end{aligned}\right] $ 与向量 $ 	\left[ \begin{aligned}-1 \\\\ 1 \\\\ \end{aligned}\right] $ 按一定比例缩放以后相加的结果等于向量 $ 	\left[ \begin{aligned}1 \\\\ 5 \\\\ \end{aligned}\right] $

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/2dimesion.png)









三元线性方程组理解为：向量 $ \left[ \begin{matrix}2 \\\\ 4 \\\\ -2 \\\\ \end{matrix}\right] $ 与向量 $ 	\left[ \begin{matrix}1 \\\\ -6 \\\\ 7 \\\\ \end{matrix}\right] $ 与向量 $ 	\left[ \begin{matrix}1 \\\\ 0 \\\\ 2 \\\\ \end{matrix}\right] $ 按一定比例缩放以后相加的结果等于向量 $ 	\left[ \begin{matrix}5 \\\\ -2 \\\\ 9 \\\\ \end{matrix}\right] $

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/3dimension.png)











## 4个基本子空间

> **基础概念：**
> 
> 线性组合：一些向量的任意标量乘法之和
> $ span\{\boldsymbol{v_1},\boldsymbol{v_2},...,\boldsymbol{v_p}\} $：所有可以表示成 $\boldsymbol{v_1},...,\boldsymbol{v_p}$ 的线性组合的向量集合（所有可以由 $\boldsymbol{v_1},...,\boldsymbol{v_p}$ 线性表出的向量的集合）

### 列空间（column space）

$ m \times n $ 维矩阵 $ A $ 的**列空间** $ C(A) $ 是由 $ A $ 的列的所有线性组合组成的集合。
$ C(A) $ 中的一个典型向量可以写成 $ A\boldsymbol{x} $ 的形式，其中 $ \boldsymbol{x} $ 为某向量，因为 $ A\boldsymbol{x} $ 表示 $ A $ 的列向量的线性组合。即$ C(A) $ 是线性变换 $ \boldsymbol{x} \mapsto A\boldsymbol{x} $ 的值域。


- $ C(A) $ 是 $\mathbb{R}^m$ 的一个子空间
- $ A $的一个最大线性无关向量组就是 $ C(A) $ 的一组基。

> **例：**
矩阵$ A＝ \left[ \begin{matrix} 1 & 0 \\\\ 4 & 3 \\\\ 2 & 3 \end{matrix}\right] $ 的列空间 $ C(A)=span\lbrace \boldsymbol{a_1},\boldsymbol{a_2}\rbrace $ 是 $ \mathbb{R}^3 $ 的一个子空间 ，其中 $ \boldsymbol{a_1}=\left[ \begin{matrix} 1 \\\\ 4 \\\\ 2 \end{matrix}\right] $、$ \boldsymbol{a_2}=\left[ \begin{matrix} 0 \\\\ 3 \\\\ 3 \end{matrix}\right] $

### 零空间（null space）

$ m \times n $ 维矩阵 $ A $ 的零空间 $ N(A) $ 是齐次方程 $ A\boldsymbol{x}=\boldsymbol{0} $ 解的集合。  

- $ N(A) $ 是 $\mathbb{R}^n$ 的一个子空间

- 求$ N(A) $的过程就是求$ A\boldsymbol{x}=\boldsymbol{0} $ 的通解的过程

> **例：**
求矩阵 $ A=\left[ \begin{matrix} 1 & 2 & 2 & 4 \\\\ 3 & 8 & 6 & 16 \\\\ \end{matrix}\right] $ 的零空间，即先求$ \left[ \begin{matrix} 1 & 2 & 2 & 4 \\\\ 3 & 8 & 6 & 16 \\\\ \end{matrix}\right]\boldsymbol{x}= \left[ \begin{matrix} 0 \\\\ 0\end{matrix} \right]$ 的通解。
将 $ A $ 化为阶梯型矩阵$ \left[ \begin{matrix} 1 & 0 & 2 & 0 \\\\ 0 & 1 & 0 & 2 \\\\ \end{matrix}\right] $，即$ \left\lbrace\begin{align\*} x\_1 &=-2x\_3 \\\\ x\_2 &= -2x\_4 \\\\ \end{align\*} \right.$，可得通解：
$$ \left[ \begin{matrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{matrix}\right]=c_1\left[ \begin{matrix}-2\\0\\1\\0 \end{matrix}\right] + c_2\left[ \begin{matrix}0\\-2\\0\\1 \end{matrix}\right] $$
所以，零空间 $ N(A) $ 为 $ \mathbb{R}^4 $ 的一个子空间
为
$ N(A)=span\{\boldsymbol{v_1},\boldsymbol{v_2}\} $，其中$ \boldsymbol{v_1}=\left[ \begin{matrix}-2\\\\0\\\\1\\\\0 \end{matrix}\right] $、$ \boldsymbol{v_1}=\left[ \begin{matrix}0\\\\-2\\\\0\\\\1 \end{matrix}\right] $
$ N(A) $ 为 $ \mathbb{R}^4 $ 的一个子空间


### 行空间（row space）

$ m \times n $ 维矩阵 $ A $ 的**行空间** $ C(A^T) $ 是由 $ A $ 的行的所有线性组合组成的集合。

- $ C(A^T) $ 是 $\mathbb{R}^n$ 的一个子空间

### 左零空间（left null space）

$ m \times n $ 维矩阵 $ A $ 的左零空间 $ N(A^T) $ 是齐次方程 $ A^T\boldsymbol{x}=\boldsymbol{0} $ 解的集合。 

- $ N(A^T) $ 是 $\mathbb{R}^m$ 的一个子空间

### 4个子空间的关系

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/4spaces.png)







- 左邻空间的向量与列空间的向量相垂直（内积为零）
- 左邻空间与列空间交于向量 $ \boldsymbol{0} $
- 列空间的维数（最大线性无关的向量个数、秩、基的个数）为 $ r $，左邻空间的维数为 $ m-r $
- 零空间的向量与行空间的向量相垂直
- 行空间的维数（最大线性无关的向量个数、秩、基的个数）为 $ r $（矩阵行向量的秩等于列向量的秩），零空间的维数为 $ n-r $



### 利用子空间判断线性方程组的解

$ A\boldsymbol{x}=b $ 的解

- 只有唯一解，则 $ \boldsymbol{b} \in C(A) $，且 $ N(A) $ 的维数是0
- 有无穷多个解，$ \boldsymbol{b} \in C(A) $，且 $N(A)$ 的维数大于0
- 无解，则$ \boldsymbol{b} \notin C(A) $


## 特征分解

### 特征值与特征向量

> **定义：**
> 设 $A$ 是 $n$ 阶矩阵，如果数 $\lambda$ 和 $n$ 维非零列向量 $\boldsymbol{x}$ 使关系式
>$$A\boldsymbol{x}=\lambda\boldsymbol{x}$$
>成立，那么，这样的数 $\lambda$ 成为矩阵 $A$ 的**特征值**，非零向量 $\boldsymbol{x}$ 称为 $A$ 的对应于特征值 $\lambda$ 的**特征向量**。

### $ A\boldsymbol{x}=\lambda\boldsymbol{x} $ 的几何意义

变换 $ \boldsymbol{x}\mapsto A\boldsymbol{x} $ 与变换 $ \boldsymbol{x}\mapsto\lambda\boldsymbol{x} $ 是等价的。即，经过变换 $ \boldsymbol{x}\mapsto A\boldsymbol{x} $ 后的向量与 $\boldsymbol{x}$ 向量共线（方向相同或相反）。

> **例：**
给定矩阵 $ A=\left[\begin{matrix} 4&1\\1&4 \end{matrix}\right] $，
对 $ \boldsymbol{x_1}=\left[\begin{matrix} 1\\0\end {matrix}\right] $，有 $ A\boldsymbol{x} =\left[\begin{matrix} 4\\1\end {matrix}\right] $；
对 $ \boldsymbol{x_2}=\left[\begin{matrix} 0\\1\end {matrix}\right] $，有 $ A\boldsymbol{x}=\left[ \begin{matrix} 1\\4\end {matrix}\right] $；
对 $ \boldsymbol{x_3}=\left[\begin{matrix} 1\\1\end {matrix}\right] $，有 $ A\boldsymbol{x}=5\left[ \begin{matrix} 1\\1\end {matrix}\right] =5\boldsymbol{x_3}$。
>![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/eigen_value.png)


### 特征分解

#### 一般矩阵

**相似矩阵：** $n$ 阶矩阵 $A$、$B$，若有可逆矩阵 $P$，使
$$P^{-1}AP=B$$
则称 $B$ 是 $A$ 的相似矩阵。

 **对角化定理：**  $n$ 阶矩阵 $A$ 与对角矩阵 $\Lambda$ 相似（即 $A$ 能对角化）的充分必要条件是 $A$ 有 $n$ 个线性无关的特征向量。

**推论：** 如果 $n$ 阶矩阵 $A$ 的 $n$ 个特征值互不相等，则 $A$ 与对角矩阵相似。

>**证明：**
>假设存在可逆矩阵 $P$，使 $P^{-1}AP=\Lambda$ 为对角矩阵，$P$ 用其列向量表示为
>$$P=(\boldsymbol{p_1},\boldsymbol{p_2},...,\boldsymbol{p_n})$$
>由 $P^{-1}AP=\Lambda$，有 $AP=P\Lambda$，即
>$$
\begin{aligned}
A(\boldsymbol{p_1},\boldsymbol{p_2},...,\boldsymbol{p_n}) &= 
(\boldsymbol{p_1},\boldsymbol{p_2},...,\boldsymbol{p_n})
\left[\begin{matrix}
\lambda_1\\&\lambda_2\\&&\ddots\\&&&\lambda_n
\end{matrix}\right] \\ &=(\lambda_1\boldsymbol{p_1},\lambda_2\boldsymbol{p_2},...,\lambda_n\boldsymbol{p_n})
\end{aligned}
$$
>于是
>$$A\boldsymbol{p}_i=\lambda_i\boldsymbol{p_i} (i=1,2,...,n)$$
>可见 $\lambda_i$ 是$A$ 的特征值，而 $P$ 的列向量 $\boldsymbol{p_i}$ 就是 $A$ 的对应于特征值 $\lambda_i$ 的特征向量。
>而且，因为P可逆，所以 $\boldsymbol{p_1},\boldsymbol{p_2},...,\boldsymbol{p_n}$ 线性无关。

#### 对称矩阵

对 $n\times n$ 的对称矩阵 $A$：
- $A$ 特征值为实数。且有 $n$ 个特征值（包含重复的特征值）
- $A$ 的不同特征值对应的特征向量相互正交（$\boldsymbol{p_1}^T\boldsymbol{p_2}=0$）
- 秩$r=Rank(A)\leq n$，即$$\underbrace{ |\lambda_1|\geq |\lambda_2|\geq ...\geq|\lambda_r|}_{r}>\underbrace{\lambda_{r+1}=...=\lambda_n}_{n-r}=0$$
- $Rank(A^TA)=Rank(AA^T)=Rank(A)=Rank(\Lambda)$
- $A$ 可正交对角化

> **证明不同特征值（$\lambda_1\neq \lambda_2$）对应的特征向量正交（$\boldsymbol{p_1^T\boldsymbol{p_2}}=0$）：**
> 因 $A$ 对称，故$\lambda_1\boldsymbol{p_1^T}=(\lambda_1\boldsymbol{p_1})^T= (A\boldsymbol{p_1})^T= \boldsymbol{p_1^T}A^T=\boldsymbol{p_1^T}A$，于是
> $$\lambda_1\boldsymbol{p_1^T}\boldsymbol{p_2}=\boldsymbol{p_1^T}A\boldsymbol{p_2}=\boldsymbol{p_1^T}\lambda_2\boldsymbol{p_2}=\lambda_2\boldsymbol{p_1^T}\boldsymbol{p_2}$$
> 即
> $$ (\lambda_1-\lambda_2)\boldsymbol{p_1^T\boldsymbol{p_2}}=0 $$
> 但 $\lambda_1\neq \lambda_2$，所以 $\boldsymbol{p_1^T\boldsymbol{p_2}}=0$，即 $\boldsymbol{p_1}$ 与 $\boldsymbol{p_2}$ 正交。

**谱分解：**

对 $n$ 阶对称矩阵 $A$，必有正交矩阵 $P$，使 $P^{-1}AP=P^TAP=\Lambda$：
$$ \begin{align*}
A&=P\Lambda P^T  \\
&=[\boldsymbol{p_1},\boldsymbol{p_2},\cdots,\boldsymbol{p_n}] \left[\begin{matrix}\lambda_1\\& \lambda_2\\&&\ddots\\&&& \lambda_n \end{matrix}\right]\left[\begin{matrix}\boldsymbol{p_1^T} \\\boldsymbol{p_2^T}\\\vdots\\\boldsymbol{p_n^T}\end{matrix}\right] \\
&=\lambda_1\boldsymbol{p_1}\boldsymbol{p_1^T}+\lambda_2\boldsymbol{p_2}\boldsymbol{p_2^T}+\cdots+\lambda_n\boldsymbol{p_n}\boldsymbol{p_n^T} \\
&=\sum_{i=1}^{n}\lambda_i\boldsymbol{p_i}\boldsymbol{p_i^T}
\end{align*}$$

上述式子分解为A的谱（特征值）确定的小块，所以称为谱分解。


### 二次型（Quadratic Form）
对 $n\times n$ 阶的对称矩阵 $A$，函数
$$ f(\boldsymbol{x})=\sum_{i,j=1}^{n} a_{ij}x_ix_j=\boldsymbol{x^T}A\boldsymbol{x} $$
被称为二次型。

> **通过函数的二次型推导矩阵的二次型：**
> 二次齐次函数
$$\begin{align*}
f(x_1,x_2,\cdots,x_n)&=a_{11}x_1^2+a_{22}x_2^2+\cdots+a_{nn}x_n^2 \\
&\quad+2a_{12}x_1x_2+2a_{13}x_1x_3+\cdots+2a_{n-1,n}x_{n-1}x_n
\end{align*}$$
> 称为二次型（所有项全部为2次）
> 当 $j>i$ 时，取$a_{ij}=a_{ji}$，则$2a_{ij}x_ix_j=a_{ij}x_ix_j+a_{ji}x_jx_i$，于是
$$\begin{align*}
f&=a_{11}x_1^2+a_{12}x_1x_2+\cdots+a_{1n}x_1x_n \\
&\quad+a_{21}x_2x_1+a_{22}x_2^2+\cdots+a_{2n}x_2x_n \\
&\quad+\cdots \\
&\quad+a_{n1}x_nx_1+a_{n2}x_nx_2+\cdots+a_{nn}x_n^2\\
&=\sum_{i,j=1}^{n}a_{ij}x_ix_j \\
&=x_1(a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n) \\
&\quad+x_2(a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n) \\
&\quad+\cdots \\
&\quad+x_n(a_{n1}x_1+a_{n2}x_2+\cdots+a_{nn}x_n) \\
&=\left[x_1,x_2,\cdots,x_n\right]\left[\begin{matrix} a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n \\a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n \\ \vdots \\a_{n1}x_1+a_{n2}x_2+\cdots+a_{nn}x_n \end{matrix}\right] \\
&=\left[x_1,x_2,\cdots,x_n\right]\left[\begin{matrix} a_{11} & a_{12} & \cdots & a_{1n} \\a_{21}&a_{22}&\cdots&a_{2n} \\ \vdots&\vdots&&\vdots \\a_{n1}&a_{n2}&\cdots&a_{nn}\end{matrix}\right]\left[\begin{matrix} x_1\\x_2\\\vdots\\x_n\end{matrix}\right] \\
&=\boldsymbol{x^T}A\boldsymbol{x}
\end{align*}$$



### 正定二次型

设二次型 $f(\boldsymbol{x})=\boldsymbol{x^T}A\boldsymbol{x}$，如果对任何 $\boldsymbol{x}\neq \boldsymbol{0}$，都有 $f(\boldsymbol{x})>0$（显然 $f(\boldsymbol{0})=0$），则称 $f$ 为正定二次型，并称对称矩阵 $A$ 是正定的；如果对于任何 $\boldsymbol{x}\neq \boldsymbol{0}$ 都有 $f(\boldsymbol{x})<0$，则称 $f$ 为负定二次型，并称对称矩阵 $A$ 是负定的。

- 半正定（positive semidefinite）：$f\geq 0$
- 正定（positive definite）：$f>0$
- 负定（negative definite）：$f<0$
- 不定（indefinite）

**对称矩阵 $A$ 为正定的充分必要条件是：$A$ 的特征值全为正**

### 特征分解的应用：PCA

> PCA：Principal Component Analysis 主成分分析，用于降维

先不考虑降维，先考虑如下的一个变换。
对$m\times n$矩阵 $X$，我们希望经过一个变换 $Y=QX$ （$Q\in \mathbb{R}^{m\times m}$）使 $Y$ 的**协方差矩阵为对角阵**。协方差矩阵为对角阵意味着行向量之间的协方差为0，而每个行向量的方差尽可能大。

$X$ 的协方差矩阵为：
$$ C_X=\frac{1}{n}XX^T $$
例如
$$ X=\left[\begin{matrix}a_1&a_2&\cdots&a_n \\
b_1&b_2&\cdots&b_n\end{matrix}\right] $$
时，协方差矩阵为：
$$ C_X=\frac{1}{n}\left[\begin{matrix}a_1&a_2&\cdots&a_n\\
b_1&b_2&\cdots&b_n\end{matrix}\right] 
\left[\begin{matrix}a_1&b_1\\
a_2&b_2\\\vdots&\vdots\\a_n&b_n\end{matrix}\right]
=\left[\begin{matrix}
\frac{1}{n}\sum_{i=1}^{n} a_i^2 & \frac{1}{n}\sum_{i=1}^{n} a_ib_i \\
\frac{1}{n}\sum_{i=1}^{n} a_ib_i & \frac{1}{n}\sum_{i=1}^{n} b_i^2
\end{matrix}\right]$$
所以，有 $Y$ 的协方差矩阵：
$$C_Y=\frac{1}{n}YY^T=\frac{1}{n}(QX)(QX)^T=\frac{1}{n}QXX^TQ^T=QC_XQ^T$$
我们的目的就是求 $Q$ 使 $C_Y$ 为对角阵。利用特征分解，我们可以将对称矩阵 $C_X$ 对角化：
$$ P^{-1}C_XP=P^TC_XP=\Lambda$$
其中，$P$ 的列向量为 $C_X$ 的特征向量，$\Lambda$是 $C_X$ 的特征值组成的矩阵。 结合上面两个式子，显然有
$$Q=P^{-1}=P^T$$

> **示例**
> $$X=\left[\begin{matrix}-1&-1&0&2&0 \\ -2&0&0&1&1\end{matrix}\right]$$
> 可求得
> $$C_X=\left[\begin{matrix} \frac{6}{5}& \frac{4}{5} \\ \frac{4}{5} & \frac{6}{5} \end{matrix}\right]$$
> 计算 $C_X$ 的特征值 $\lambda_1$、$\lambda_2$，特征向量 $\boldsymbol{p_1}$、$\boldsymbol{p_2}$，得
> $$ \begin{align*}C_X&= P\Lambda P^T  \\
&=[\boldsymbol{p_1},\boldsymbol{p_2}] \left[\begin{matrix}\lambda_1\\& \lambda_2 \end{matrix}\right]\left[\begin{matrix}\boldsymbol{p_1^T} \\\boldsymbol{p_2^T}\end{matrix}\right] \\
&=\left[\begin{matrix} \frac{1}{\sqrt{2}} &-\frac{1}{\sqrt{2}}\\ \frac{1}{\sqrt{2}}&\frac{1}{\sqrt{2}}  \end{matrix}\right] \left[\begin{matrix}2 &\\ &\frac{2}{5} \end{matrix}\right] \left[\begin{matrix} \frac{1}{\sqrt{2}} &\frac{1}{\sqrt{2}}\\ -\frac{1}{\sqrt{2}}&\frac{1}{\sqrt{2}}  \end{matrix}\right]
\end{align*} $$
> 因此
> $$\begin{align*}Y&=QX=P^TX \\
&= \left[\begin{matrix}\boldsymbol{p_1^T}\\\boldsymbol{p_2^T} \end{matrix}\right]X\\
&=\left[\begin{matrix} \frac{1}{\sqrt{2}} &\frac{1}{\sqrt{2}}\\ -\frac{1}{\sqrt{2}}&\frac{1}{\sqrt{2}}  \end{matrix}\right] \left[\begin{matrix}-1&-1&0&2&0 \\ -2&0&0&1&1\end{matrix}\right] \\
&=\left[\begin{matrix}-\frac{3}{\sqrt{2}}&-\frac{1}{\sqrt{2}}&0&\frac{3}{\sqrt{2}}&\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} &0&-\frac{1}{\sqrt{2}}& \frac{1}{\sqrt{2}}\end{matrix}\right]
\end{align*}$$
> 可以观察到Y中的每一行，都是X的一个特征向量的转置与X相乘的结果，即 $Y=\left[\begin{matrix}y_1^T\\y_2^T\end{matrix}\right]=\left[\begin{matrix}\boldsymbol{p_1^T}X\\ \boldsymbol{p_2^T}X\end{matrix}\right]$

再考虑降维。
上面例子中$C_X$ 的两个特征值 $\lambda_1=2$、$\lambda_2=\frac{2}{5}$，相对来说 $\lambda_1$ 比 $\lambda_2$大不少，而
$$\begin{align*}C_X= P\Lambda P^T  
=[\boldsymbol{p_1},\boldsymbol{p_2}] \left[\begin{matrix}\lambda_1\\& \lambda_2 \end{matrix}\right]\left[\begin{matrix}\boldsymbol{p_1^T} \\\boldsymbol{p_2^T}\end{matrix}\right] 
=\lambda_1\boldsymbol{p_1}\boldsymbol{p_1^T}+\lambda_2\boldsymbol{p_2}\boldsymbol{p_2^T}
\end{align*}$$
因此，我们可以只取等式最右边第一项 $ \lambda_1\boldsymbol{p_1}\boldsymbol{p_1^T} $ 来近似描述 $ C_X $，即 $ P=[\boldsymbol{p_1}] $。进一步地，
$$
\begin{align*}Y&=QX=P^TX=\boldsymbol{p_1^T}X \\
&=\left[\begin{matrix} \frac{1}{\sqrt{2}} &\frac{1}{\sqrt{2}}  \end{matrix}\right] \left[\begin{matrix}-1&-1&0&2&0 \\ -2&0&0&1&1\end{matrix}\right] \\
&=\left[\begin{matrix}-\frac{3}{\sqrt{2}}&-\frac{1}{\sqrt{2}}&0&\frac{3}{\sqrt{2}}&\frac{1}{\sqrt{2}}\end{matrix}\right]
\end{align*}
$$
从而达到了降维的目的。


因为之前没有接触PCA，这里只按照程博的讲解纪录了降维的原理和步骤，实际运用中肯定有很大不同（程博的讲解也提到了实际中PCA可能不是用特征分解而是SVD，因为特征分解的性质不好等balabala），总结PCA的要点：

- KL变换来的，本质是把协方差矩阵对角化。
- 对角化：使不同行向量间的协方差为0，每个行向量的方差尽可能大。

## SVD

> Singular Value Decomposition 奇异值分解

![Alt text](http://7qn7rt.com1.z0.glb.clouddn.com/ml/matrix/svd.png)















## 矩阵导数

- $Y=AX$，$\frac{D(Y)}{D(X)} = A^T$
- $Y=XA$，$\frac{D(Y)}{D(X)} = A$
- $Y=A^TXB$，$\frac{D(Y)}{D(X)} = AB^T$
- $Y=A^TX^TB$，$\frac{D(Y)}{D(X)} = BA^T$
- $\frac{D(Y^T)}{D(X)} = \frac{D(Y)}{D(X)} $

## 参考资料

- 七月算法机器学习课程
- 矩阵分析与应用 张贤达