title: 机器学习小白的Kaggle学习：信用卡诈骗识别
date: 2017-06-25 13:59
categories: machine_learning
tags: [机器学习,Kaggle,逻辑回归]
description: 理论是枯燥的，实践是艰难的，一个机器学习领域的新手在有一点理论基础后该如何一步步深入实践呢？已经有很多大神建议我们好好利用Kaggle及其类似的学习、竞赛平台，既然我是小白，那么就老老实实接受大神们的建议吧。这篇文章就是对Kaggle上Credit Card Fraud Detection这份数据集的学习。
---


理论是枯燥的，实践是艰难的，一个机器学习领域的新手在有一点理论基础后该如何一步步深入实践呢？已经有很多大神建议我们好好利用Kaggle及其类似的学习、竞赛平台，既然我是小白，那么就老老实实接受大神们的建议吧。这篇文章就是对Kaggle上[Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud)这份数据集的学习。

其实，我下载了数据集之后也是束手无策的，别说不知道怎么去调用sklearn包里的函数来建模，就是利用NumPy和Pandas来对数据进行处理、分析也要去查查什么操作该用什么函数。所以，既然是学习，而且是刚刚开始的学习，就不要想着完全依靠自己来完成了。更厚颜无耻地，我决定从阅读Kaggle上这个数据集里[点赞数最高的kernel](https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now)开始。这篇文章的内容基本来源于这个kernel，我所做的工作可能主要在于翻译和整理成文章的形式，此外，有一些基本知识的补充。

## 数据集简介

这是一份欧洲信用卡使用者的消费数据，284807条记录中有492条消费属于信用卡诈骗，因此这是一份极度不均衡的数据。

数据已经经过了PCA降维，由于保密的原因，原始的特征信息和背景信息已经基本被干掉了，只剩下名字为V1到V28的28个特征和Time、Amount。Time可以理解为消费时间，但是是相对的，Amount是消费金额。

每条记录有一个Class，表示是否是信用卡诈骗，1表示诈骗，0表示正常消费。

使用Pandas加载数据，然后看看数据到底长啥样：
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv("input/creditcard.csv")
data.head()
```
输出结果：

|   -   | Time |  V1   | V2    |  ...  |  V28 | Amount | Class |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| 0    |   0.0 |  -1.359807  | -0.072781 | ... | -0.021053 | 149.62 | 0 |
| 1    |   0.0 |  1.191857 | 0.266151 | ... | 0.014724 | 2.69 | 0 |
| 2    |   1.0 |  -1.358354  | -1.340163 | ... | -0.059752 | 378.66 | 0 |
| 3    |   1.0 |  -0.966272  | -0.185226 | ... | -0.061458 | 123.50 | 0 |
| 4    |   2.0 |  -1.158233  | 0.877737 | ... | 0.215153 | 69.99 | 0 |

## 分析

对类别进行一个统计，看看数据的不均衡程度：
``` python
count_class = pd.value_counts(data['Class'], sort=True).sort_index()
print(count_class)
```
可以看到确实只有492条的Class为1（属于诈骗），数据非常不均衡。
> 0    284315
> 1       492
> Name: Class, dtype: int64

对于数据不均衡问题，一般有如下处理方法：
- 收集更多数据。显然这里不可行
- 重采样，使不同分类数据的比例接近
  - 过采样，对数据占比少的分类的数据进行copy
  - 欠采样，从数据占比多的分类中选取部分数据来使用
- 更改性能指标
  - F1score
  - 代价敏感学习
  - Precision、Recall等

## 处理方法

1. 一般情况下是需要进行特征处理的，但这份数据的特征已经处理过了，所以这一步不需要了。
2. 对数据进行欠采样，并基于逻辑回归来比较欠采样和不进行欠采样的效果。
3. 使用Precision、Recall、ROC等指标来评估模型。


## 数据准备

虽然无需特征处理，但也需要进行一些简单的数据处理，例如Amount特征标准化、重采样、训练集数据集划分等。

### 规范化Amount特征

``` python
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)
```

### 将数据拆分成特征和标签

``` python
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']
```

### 欠采样

这里使用最简单的欠采样的方法来得到一份分布均衡的样本：有多少欺诈交易就从正常交易中随机选择多少条记录。
有一个叫做SMOTE的过采样方法很流行，如果要过采样的话可以采用，这里暂不使用。

``` python
# 少数类（欺诈交易）的数量，这些交易的index
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# 从正常交易的index中 随机选出 跟欺诈交易数量相等的index
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)

# 要采样数据的index
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# 欠采样得到的数据
under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))
```
输出如下：
> Percentage of normal transactions:  0.5
> Percentage of fraud transactions:  0.5
> Total number of transactions in resampled data:  984

### 划分数据集、测试集

``` python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions train dataset: ", X_train.shape[0])
print("Number transactions test dataset", X_test.shape[0])
print("Total number of transactions: ", len(X_train) + len(X_test))

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = \
train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)

print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample) + len(X_test_undersample))
```

输出如下：

> Number transactions train dataset:  199364
> Number transactions test dataset 85443
> Total number of transactions:  284807
>
> Number transactions train dataset:  688
> Number transactions test dataset:  296
> Total number of transactions:  984

## 基于欠采样数据的学习

在开始看代码之前首先要了解几个基本概念：真正例（True Positive）、假正例（False Positive）、真反例（True Negative）、假反例（False Negative）。对二分类问题，根据样本的真实标签和预测结果可以划分如下：

<table align="center">
<thead>
<tr><th rowspan="2">真实情况</th><th colspan="2" align="center">预测结果</th></tr>
<tr><th align="center">正例</th><th align="center">反例</th></tr>
</thead>
<tbody>
<tr><td align="center">正例</td><td>真正例（TP）</td><td>假反例（FN）</td></tr>
<tr><td align="center">反例</td><td>假正例（FP）</td><td>真反例（TN）</td></tr>
</tbody>
</table>

还需要了解几个性能指标：精度（Accuracy）、查准率（Precision，又称准确率）、查全率（Recall，又称召回率）。

- Accuracy = (TP + TN)  / total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

对Precision和Recall通俗的理解：

- Precision：判别为正例的结果中真正例的比率
- Recall：所有正例中，判别结果是正例的比率

也需要了解一个测试方法：k折交叉验证法（k-fold cross validation）。将数据集D划分为k个大小相同的互斥的子集，即 $D = D_1 \cup D_2 \cup \dots \cup D_k, D_i \cap D_j = \varnothing (i \neq j) $ 。每个子集 $D_i$ 都尽可能保持数据分布的一致性，即从D中通过分层采样得到。然后，每次用k-1个子集的并集作为训练集，余下的那个子集作为测试集；这样就可获得k组训练/测试集，从而可以进行k此训练和测试，最终返回的是这k次测试结果的均值。

在后面我们绘制ROC曲线、P-R曲线时也会对它们进行简单介绍。如果想对这些概念进行进一步了解，建议阅读周志华老师的《机器学习》第二章“模型评估与选择”。

这里使用了“逻辑回归”这个机器学习方法，所以还需要对逻辑回归先有一个了解。可以阅读周志华老师的《机器学习》第三章“线性模型”，3.3节的“对数几率回归”就是逻辑回归。
也可以参考我之前的学习笔记[机器学习之线性回归](http://coderss.me/2016/11/10/machine_learning/linear-regression/)、[机器学习之逻辑回归](http://coderss.me/2016/11/27/machine_learning/logistic-regression/)。

下面的代码先实现了一个函数：通过k折交叉验证（这里k=5），利用逻辑回归模型，在给定数据集的基础上，从一组给定的模型参数（C_param）中，选出最优的模型参数。考虑到实际场景，我们希望尽可能找出信用卡欺诈的消费，所以，这里“最优”的判断使用的参数是召回率。

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, \
recall_score, classification_report, auc

def print_Kfold_scores(X_train_data, y_train_data):
    fold = KFold(len(y_train_data), 5, shuffle=False)
    
    # 候选C_param
    c_param_range = [0.01, 0.1, 1, 10, 100]
    
    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    result_table['C_parameter'] = c_param_range
    
    j = 0
    for c_param in c_param_range:
        print('=====================================')
        print('C parameter: ', c_param)
        print('-------------------------------------')
        print('')
        
        recall_accs = []
        # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
        for iteration, indices in enumerate(fold, start=1):
            # 建立逻辑回归模型
            lr = LogisticRegression(C = c_param, penalty='l1')
            # 训练
            lr.fit(X_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            # 预测
            y_pred_undersample = lr.predict(X_train_data.iloc[indices[1], :].values)
            
            # 召回率计算
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            
            print('Iteration ', iteration, ' recall score: ', recall_acc)
            
        result_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    # 召回率最高的C_param
    best_c = result_table.loc[result_table['Mean recall score'].idxmax()]['C_parameter']
    
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c
```

在欠采样的数据集上调用函数`print_Kfold_scores`:
```python
best_c = print_Kfold_scores(X_train_undersample, y_train_undersample)
```
输出如下：
> =====================================
>C parameter:  0.01
>\-------------------------------------
>
>Iteration  1  recall score:  0.931506849315
>Iteration  2  recall score:  0.917808219178
>Iteration  3  recall score:  1.0
>Iteration  4  recall score:  0.959459459459
>Iteration  5  recall score:  0.954545454545
>
>Mean recall score  0.9526639965
>
>=====================================
>C parameter:  0.1
>\-------------------------------------
>
>Iteration  1  recall score:  0.849315068493
>Iteration  2  recall score:  0.86301369863
>Iteration  3  recall score:  0.932203389831
>Iteration  4  recall score:  0.945945945946
>Iteration  5  recall score:  0.909090909091
>
>Mean recall score  0.899913802398
>
>=====================================
>C parameter:  1
>\-------------------------------------
>
>Iteration  1  recall score:  0.849315068493
>Iteration  2  recall score:  0.904109589041
>Iteration  3  recall score:  0.983050847458
>Iteration  4  recall score:  0.945945945946
>Iteration  5  recall score:  0.924242424242
>
>Mean recall score  0.921332775036
>
>=====================================
>C parameter:  10
>\-------------------------------------
>
>Iteration  1  recall score:  0.86301369863
>Iteration  2  recall score:  0.890410958904
>Iteration  3  recall score:  0.983050847458
>Iteration  4  recall score:  0.945945945946
>Iteration  5  recall score:  0.924242424242
>
>Mean recall score  0.921332775036
>
>=====================================
>C parameter:  100
>\-------------------------------------
>
>Iteration  1  recall score:  0.86301369863
>Iteration  2  recall score:  0.890410958904
>Iteration  3  recall score:  0.983050847458
>Iteration  4  recall score:  0.945945945946
>Iteration  5  recall score:  0.924242424242
>
>Mean recall score  0.921332775036
>
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
>Best model to choose from cross validation is with C parameter =  0.01
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

可以发现，逻辑回归模型的参数C取0.01时召回率最高。这就是我们基于欠采样数据学习得到的模型：参数C=0.01的逻辑回归模型。

## 模型在测试集上的表现

接下来，我们需要看看上面学习得到的模型在测试集上的表现。混淆矩阵可以对分类结果给出一个直观的认识，我们先实现混淆矩阵的绘制函数：

```python
import itertools

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title="Confusion matrix",
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap) #interpolation：插值
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0) # x刻度 第一个参数是location，第二个参数是label
    plt.yticks(tick_marks, classes) # y刻度
    
    # 归一化，这里不需要使用
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        1
    
    # 给每个区域写具体的值
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black") 
    
    plt.tight_layout() # Automatically adjust subplot parameters to give specified padding.
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
```

首先，看看模型在欠采样数据的测试集上的表现：

``` python
# 逻辑回归，C_parameter使用上面比较得到的最优参数(0.01)。
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel()) #训练数据：欠采样的训练集
y_pred_undersample = lr.predict(X_test_undersample.values) #预测数据：欠采样的测试集

# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 绘制 non-normalized 的混淆矩阵
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
```

输出结果如下：
> Recall metric in the testing dataset:  0.931972789116
> ![confusion matrix plot](http://7qn7rt.com1.z0.glb.clouddn.com/ml/kaggle/credit_fruad_confusion1.png)

可以看到模型在欠采样数据集上的召回率达到了93.2%，作为一个简单的模型来说已经相当不错啦，那进一步来看看模型在全部数据的测试集上的表现：

``` python
# 逻辑回归，C_parameter使用上面比较得到的最优参数。
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel()) #训练数据：欠采样的训练集
y_pred = lr.predict(X_test.values) #预测数据：所有数据的测试集

# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 绘制 non-normalized 的混淆矩阵
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
```
输出结果如下：
> Recall metric in the testing dataset:  0.918367346939
> ![confusion matrix plot](http://7qn7rt.com1.z0.glb.clouddn.com/ml/kaggle/credit_fruad_confusion2.png)

可以看到召回率还是比较理想的，但是准确率就不太好了。




## ROC曲线和P-R曲线

作为学习，我们看看ROC曲线和Precision-Recall曲线的绘制。这里给出它们的介绍和绘制代码、结果，至于这两个曲线具体的作用留着读者自己去感受吧。

ROC全称是“受试者工作特征”（Receiver Operating Characteristic）曲线。我们根据学习器的预测结果对样例进行排序，排在前面的是学习器认为“最可能”是正例的样本，排在最后的则是学习器认为“最不可能”是正例的样本。按此顺序逐个把样本作为正例进行预测，每次计算出两个重要量的值，分别以它们为横、纵坐标作图，就得到了“ROC曲线”。纵轴是“真正例率”（True Positive Rate，TPR），横轴是“假正例率”（False Positive Rate，FPR）：

- TPR = TP / (TP + FN)
- FPR = FP / (TN + FP)

我个人对“真正例率”和“假正例率”通俗的理解：

- 真正例率：所有正例中，判别结果是正例的比率
- 假正例率：所有反例中，判别结果是正例的比率

AUC（Area Under ROC Curve）：ROC曲线下的面积。

同样的，我们根据学习器的结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次可以计算出当前的查全率（Recall）、查准率（Precision）。以Precision为纵轴、Recall为横轴作图，就得到了Precision-Recall曲线，简称P-R曲线。

以下为绘制ROC曲线的代码：
``` python
# 回归模型
lr = LogisticRegression(C = best_c, penalty='l1')
# decision_function(X): Predict confidence scores for samples.
# 模型学习、预测每个样本的得分
y_pred_undersample_score = lr.fit(X_train_undersample, y_train_undersample.values.ravel())\
.decision_function(X_test_undersample.values)

# 计算fpr、tpr
fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(), y_pred_undersample_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```
下图为绘制得到的ROC曲线，对比以下对应于随机猜测模型的虚线，效果还是不错的。
![ROC Cruve](http://7qn7rt.com1.z0.glb.clouddn.com/ml/kaggle/credit_fruad_roc.png)

以下为绘制P-R曲线的代码：
``` python
# 计算precision和recall
precision, recall, threshold = precision_recall_curve(y_test_undersample.values.ravel(), y_pred_undersample_score)

# 绘制P-R曲线
plt.title('Precision-Recall Curve')
plt.plot(recall, precision)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.10])
plt.ylim([0.0, 1.10])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```
绘制得到的P-R曲线：
![Precison-Recall Cruve](http://7qn7rt.com1.z0.glb.clouddn.com/ml/kaggle/credit_fruad_pr.png)

## 结语

作为一个小白的初次kaggle体验到这里就准备结束了，原[kernel](https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now)中还有对整个数据集使用同样方法来训练的探索。不管怎样，还是强烈建议大家花时间去阅读原kernel，写的确实挺好。
