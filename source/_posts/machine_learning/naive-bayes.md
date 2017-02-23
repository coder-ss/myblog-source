title: 机器学习之朴素贝叶斯
date: 2016-12-06 17:22
categories: machine_learning
tags: [机器学习,朴素贝叶斯]
description: 这篇笔记先对生成式学习方法和判别式学习方法做了介绍，然后介绍了朴素贝叶斯假设，之后介绍了朴素贝叶斯假设在二分类问题上的应用和拉普拉斯平滑，最后对垃圾邮件分类问题基于Python进行了实践。
---

## 生成式与判别式

监督学习的任务就是要学习到一个模型，对于给定一个输入 $\boldsymbol{x} = \left[ \begin{matrix} x_1 & x_2 & \cdots & x_n \end{matrix} \right]^{\text{T}} $，能够预测其输出 $y$。这个模型一般形式为决策函数：
$$ y = h_\boldsymbol{\theta}\left(\boldsymbol{x}\right) $$
或者条件分布概率：
$$p\left(y\mid\boldsymbol{x}\right)$$

**判别式学习方法**（discriminative learning algorithms）由数据直接学习到决策函数 $h_\boldsymbol{\theta}\left(\boldsymbol{x}\right) $ 或者条件概率 $p\left(y\mid\boldsymbol{x}\right)$ 分布作为预测的模型。

**生成式学习方法**（generative learning algorithms）则是对条件概率 $p\left(\boldsymbol{x}\mid y\right)$（和概率 $p\left(y\right)$）建模，然后依据贝叶斯公式可以计算出给定 $\boldsymbol{x}$ 下 $y$ 的概率：
$$ p\left(y\mid \boldsymbol{x}\right) = \frac{p\left(\boldsymbol{x}\mid y\right)p\left(y\right)}{p\left(\boldsymbol{x}\right)} $$
上式中 $p\left( \boldsymbol{x} \right)$ 可以求解出来，例如对于二分类问题：
$$p\left(\boldsymbol{x}\right) = p\left(\boldsymbol{x}\mid y=1\right)p\left(y=1\right) + p\left(\boldsymbol{x}\mid y=0\right)p\left(y=0\right)$$
事实上，对于预测问题，我们可以不用求解出 $p\left( \boldsymbol{x} \right)$：
$$\begin{align*}
\arg \max_y p\left(y\mid \boldsymbol{x}\right) &= \arg \max_y \frac{p\left( \boldsymbol{x}\mid y \right)p\left(y\right)}{p\left( \boldsymbol{x} \right)} \\ 
&= \arg \max_y p\left( \boldsymbol{x}\mid y \right)p\left(y\right)
\end{align*}$$

生成式学习方法特点：
- 可以还原出联合概率分布 $p\left( \boldsymbol{x}, y \right) = p\left( \boldsymbol{x}\mid y \right)p\left( y \right)$
- 收敛速度更快，当样本容量增加时，学到的模型可以更快地收敛于真实模型
- 存在隐变量时，仍然可以用生成式学习方法，此时判别式学习方法就不能用

判别式学习方法特点：
- 直接面对预测，往往学习的准确率更高
- 可以对数据进行各种程度上的抽象、定义特征并使用特征，因此可以简化学习问题

## 朴素贝叶斯

上面介绍生成式算法时我们已经利用贝叶斯公式建立了 $p\left( y\mid \boldsymbol{x} \right)$ 与 $p\left( \boldsymbol{x}\mid y \right)$ 的关系，为了方便求解 $p\left( \boldsymbol{x}\mid y \right)$，我们假设 $x_i$ 在给定 $y$ 的条件下是独立的，这个假设称为**贝叶斯假设**（Naive Bayes assumption），也叫**属性条件独立性假设**（attribute conditional independence assumption）。

通过朴素贝叶斯假设我们可以简化 $p\left( \boldsymbol{x}\mid y \right)$ 的计算：
$$\begin{align*} 
p\left( \boldsymbol{x} \mid y \right) &= p\left( x_1, x_2, \ldots, x_n \mid y \right) \\ 
&= p\left( x_1 \mid y \right) p\left( x_2 \mid y, x_1 \right) \cdots p\left( x_n \mid y, x_1, x_2, \ldots, x_{n-1} \right) \\
&= p\left( x_1 \mid y \right) p\left( x_2 \mid y \right) \cdots p\left( x_n \mid y \right) \\
&= \prod_{i=1}^{n} p\left( x_i \mid y \right)
\end{align*}$$
上述推导中的第二行到第三行用到了朴素贝叶斯假设。

考虑垃圾邮件识别问题，假定一封垃圾邮件样本中出现了“买”、“价格”两个词，朴素贝叶斯假设的意思是：就算知道“买”已经出现在了这封邮件中，也对“价格”是否出现在这封邮件中没有影响。显然这个假设不太科学，但在实际中效果却非常好。

### 词集模型

对于垃圾邮件识别问题（或类似的文本处理问题），一个很常用的方式是用一个特征向量来表示一封电子邮件，特征向量的长度是词库的大小。如果一封邮件中包含词库中的第i个单词，就设置 $x_i = 1$，对于邮件中没有出现过的单词，全部设置为0。例如，向量
$$ \begin{align*} \boldsymbol{x} = \left[ \begin{matrix} 1 \\ 0  \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{matrix} \right] \quad \begin{matrix} \text{a} \\  \text{aardvark} \\ \text{aardwolf} \\ \vdots \\ \text{buy} \\ \vdots \\ \text{zygmurgy} \end{matrix} \end{align*} $$
表示该邮件中包含了“a”、“buy”，而不包含“aardvark”、“aardwolf”、“zygmurgy”。实际中，一般不会使用整个词库作为特征向量，而是只使用训练样本中出现了的词（去掉停用词），从而降低特征向量的维度、减小存储空间和计算量。

上述特征向量只考虑了单词在邮件中出现与否，而没有考虑每个单词在该邮件中出现的次数，称为**词集模型**。相应地，**词袋模型**会考虑每个单词在该邮件中出现的次数。

### 模型求解

对于垃圾邮件的识别，我们希望估计在给定邮件的情况下，邮件为垃圾邮件的概率：
$$p \left( y=1 \mid \boldsymbol{x} \right) = \frac{p\left( \boldsymbol{x} \mid y=1 \right) p\left( y=1 \right)}{p\left( \boldsymbol{x} \right)}$$
根据贝叶斯假设，有
$$ \begin{align*}
p \left( y=1 \mid \boldsymbol{x} \right) &= \frac{p\left( \boldsymbol{x} \mid y=1 \right) p\left( y=1 \right)}{p\left( \boldsymbol{x} \right)} \\ 
&= \frac{\left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=1 \right) \right) p\left( y=1 \right)} {\left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=1 \right) \right) p\left( y=1 \right) + \left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=0 \right) \right) p\left( y=0 \right)}
\end{align*} $$


显然，我们需要对 $p\left( x_i \mid y=1 \right)$、$p\left( x_i \mid y=0 \right)$、$p\left( y \right)$ 建模。


对于包含m个样本的样本集，我们可以统计出如下概率进而求解上式：
1. 垃圾邮件中词 $x_i$ 出现的概率
$$p \left( x_i =1 \mid y=1 \right) = \frac{\sum \limits_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 1 \right\}}{\sum \limits_{j=1}^{m} 1 \left\{ y^{(j)} = 1 \right\} } $$
2. 正常邮件中词 $x_i$ 出现的概率
$$p \left( x_i =1 \mid y=0 \right) = \frac{\sum \limits_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 0 \right\}}{\sum \limits_{j=1}^{m} 1 \left\{ y^{(j)} = 0 \right\} } $$
3. 垃圾邮件所占的比率
$$ p \left( y=1 \right) = \frac{\sum \limits_{j=1}^{m} 1\left\{ y^{(j)} = 1 \right\}}{m} $$

其中，$\boldsymbol{x}^{(j)}$、$y^{(j)}$ 表示第j封邮件的特征向量和标签，$x_i^{(j)}$ 表示第j封邮件中 $x_i$ 的值，$\wedge$ 表示“and”关系，$1 \left\lbrace \cdots \right\rbrace$ 是指示函数，当 $\cdots$ 为真时，指示函数的值为1，否则指示函数的值为0。


> 这里说上面三个概率是根据样本被统计出来的，事实上，这种说法不太科学。样本统计出来的是样本的参数，而不是总体的参数。
> 对于总体参数估计（parameter estimation），统计学界的两个学派分别提供了不同的解决方案：
> 1. **频率主义学派**（Frequentist）认为参数虽然未知，但却是客观存在的固定值，因此，可以通过优化似然函数等准则来确定参数值；
> 2. **贝叶斯学派**（Bayesian）则认为参数是未观察到的随机变量，其本身也可能有分布，因此，可假定参数服从一个先验分布，然后基于观测到的数据来计算参数的后验分布。
> 
>对于上面三个概率，吴恩达老师给出了依据极大似然估计的推导，因为有个别地方没理解透彻，所以没有在上面进行解释，这里直接给出吴恩达老师的推导：
>记 $\phi\_{i \mid y=1} = p \left( x\_i=1 \mid y=1 \right)$、$\phi\_{i \mid y=0} = p \left( x\_i=1 \mid y=0 \right)$、$\phi\_{y}=p\left(y=1\right)$。对于m个训练样本 $\left( x^{(j)}, y^{(j)}; j=1,2,\ldots,m \right)$，极大似然函数为：
>$$L\left( \phi_{y}, \phi_{i \mid y=0}, \phi_{i \mid y=1} \right) = \prod_{j=1}^{m} p\left( x^{(j)}, y^{(j)} \right)$$
>最大化上式就是对 $\phi\_{y}$、$\phi\_{i \mid y=0}$、$ \phi\_{i \mid y=1}$进行极大似然估计
>$$\phi_{j \mid y=1} = \frac{\sum_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 1 \right\}}{\sum_{j=1}^{m} 1 \left\{ y^{(j)} = 1 \right\} } $$
>$$\phi_{j \mid y=0} = \frac{\sum_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 0 \right\}}{\sum_{j=1}^{m} 1 \left\{ y^{(j)} = 0 \right\} } $$
>$$\phi_{y} = \frac{\sum_{j=1}^{m} 1\left\{ y^{(j)} = 1 \right\}}{m}$$











## 拉普拉斯平滑

假设现在有一封邮件需要判断是否是垃圾邮件，但邮件中有一个单词在样本中没有出现过。假设特征向量的长度为50000，而这个没有出现过的单词是第35000个单词，那么
$$p\left( x_{35000} \mid y=1 \right) = \frac{\sum_{j=1}^{m} 1 \left\{ x_{35000}^{(j)} = 1 \wedge y^{(j=1)} = 1 \right\}}{ \sum_{j=1}^{m} 1 \left\{ y^{(j)} = 1 \right\} } = 0$$
$$p\left( x_{35000} \mid y=0 \right) = \frac{\sum_{j=1}^{m} 1 \left\{ x_{35000}^{(j)} = 1 \wedge y^{(j=1)} = 0 \right\}}{ \sum_{j=1}^{m} 1 \left\{ y^{(j)} = 0 \right\} } = 0$$

所以，这封邮件是垃圾邮件的概率为
$$ \begin{align*}
p \left( y=1 \mid \boldsymbol{x} \right) 
&= \frac{\left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=1 \right) \right) p\left( y=1 \right)} {\left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=1 \right) \right) p\left( y=1 \right) + \left( \prod \limits_{i=1}^{n} p\left( x_i \mid y=0 \right) \right) p\left( y=0 \right)} \\
&= \frac{0}{0}
\end{align*} $$

因此，我们的模型在这种条件下的结果是没有意义的。

问题的根源在于我们仅仅因为有些单词在有限的训练集中没有出现过，就认为其在总体中出现的概率为0。

多项式随机变量z的取值是 $\left\lbrace 1,2,\ldots, k \right\rbrace$ ，有m个样本 $z^{(1)},\ldots,z^{(m)}$，那么
$$p\left(z=j\right) = \frac{\sum_{i=1}^{m} 1 \left\{ z^{(i)}=j \right\}}{m}$$
显然，有些概率会等于0。用**拉普拉斯平滑（Laplace smoothing）**，将上面的概率替换为
$$p\left(z=j\right) = \frac{\sum_{i=1}^{m} 1 \left\{ z^{(i)}=j \right\} + 1}{m + k}$$
这里分子加1，而分母加了k，保证总体的概率和是1，但是没有一个概率等于0。

对于 $p \left( x_i =1 \mid y=1 \right)$、$p \left( x_i =1 \mid y=0 \right)$，这里 $k=2$：
$$p \left( x_i =1 \mid y=1 \right) = \frac{\sum \limits_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 1 \right\} + 1}{\sum \limits_{j=1}^{m} 1 \left\{ y^{(j)} = 1 \right\} + 2} $$
$$p \left( x_i =1 \mid y=0 \right) = \frac{\sum \limits_{j=1}^{m} 1 \left\{ x_{i}^{(j)}=1 \wedge y^{(j)} = 0 \right\} + 1}{\sum \limits_{j=1}^{m} 1 \left\{ y^{(j)} = 0 \right\} + 2} $$


## 实践

对垃圾邮件识别问题进行了实践。代码和数据以ipython notebook的形式放在了[github](https://github.com/coder-ss/ml-learn/tree/master/naive-bayes)上。

一共有三份数据：

- [enron_email](http://csmining.org/index.php/enron-spam-datasets.html)
- [ling_email](http://csmining.org/index.php/ling-spam-datasets.html)
- [CSDMC2010_email](http://csmining.org/index.php/spam-email-datasets-.html)


首先引入依赖项、构建停用词：
``` python
import os
import sys
import re
import random
import chardet

stop_word_list = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","re","subject"]

stop_words = dict(zip(stop_word_list, [1] * len(stop_word_list)))
```

为了后续调用方便，构建以下函数：
``` python
def extract_words(text, _stop_words):
    """ 提取邮件中的单词（重复单词只提取一次）

    :param text: 邮件正文
    :param _stop_words: 停用词
    :return: 单词列表
    """
    _words = re.findall('[a-zA-Z\d]{2,}', text)
    _word_set = []
    for w in _words:
        if w in _stop_words.keys() or re.search('\d', w):
            continue
        _word_set.append(w.lower())
    _word_set = list(set(_word_set))

    return _word_set
    
    
def load_email(filename, _stop_words):
    """ 根据文件名加载邮件
    
    :param filename: 文件名
    :param _stop_words: 停用词
    :return: 邮件（用一个单词列表表示）
    """
    with open(filename, "rb") as _fp:
        ec = chardet.detect(_fp.read())['encoding']
    with open(filename, "r", encoding=ec) as _fp:
        _email = extract_words(_fp.read(), _stop_words)
    return _email

    
def load_data_set(data_name):
    """ 加载数据

    :param data_name: 数据名：enron_email, ling_bare_email, CSDMC2010_email
    :return: 正常邮件集，垃圾邮件集
    """
    _emails_ham = {}; _emails_spam = {}
    for _fn in  os.listdir('./data/%s/ham' % data_name):
        if _fn == '.DS_Store':
            continue

        _email_list = load_email('./data/%s/ham/%s' % (data_name, _fn), stop_words)
        if len(_email_list) > 5:
            _emails_ham[_fn] = _email_list

    for _fn in  os.listdir('./data/%s/spam' % data_name):
        if _fn == '.DS_Store':
            continue

        _email_list = load_email('./data/%s/spam/%s' % (data_name, _fn), stop_words)
        if len(_email_list) > 5:
            _emails_spam[_fn] = _email_list

    return _emails_ham, _emails_spam
```

加载enron_email数据
``` python
# 加载enron_email数据（较慢，可能需要1分钟）
enron_hams, enron_spams = load_data_set('enron_email')
print(len(enron_hams), len(enron_spams))
```

以enron_email数据为样本来训练，取2000个作为测试集，不参与训练。
``` python
def word_count(emails, test_fns=[]):
    """ 统计每个单词在邮件中出现的次数

    :param emails: 所有邮件
    :param test_fns: 测试用例的名字
    :return: 单词出现的次数
    """
    _word_count = {}
    for _fn in emails:
        if _fn in test_fns.keys():
            continue  # 排除测试用例
        for w in emails[_fn]:
            _word_count[w] = _word_count.get(w, 0) + 1
    return _word_count


def calc(email, _p_y1, _p_xi_y0, _p_xi_y1, _p_xi_y0_not_appear, _p_xi_y1_not_appear, is_use_not_appear=True):
    """ 计算一个邮件是否是垃圾邮件
    
    :param email: 邮件内容，由单词组成的list
    :param _p_y1: 样本中正常邮件所占的比率
    :param _p_xi_y0: 每个单词在垃圾邮件中出现的概率
    :param _p_xi_y1: 每个单词在正常邮件中出现的概率
    :param _p_xi_y0_not_appear: 垃圾邮件中未出现过的词，给一个很小的概率
    :param _p_xi_y1_not_appear: 正常邮件中未出现过的词，给一个很小的概率
    :param is_use_not_appear: 是否使用未出现过的词，True表示使用
    :return: 正常邮件与垃圾邮件概率的比值，大于1表示分类结果是正常邮件
    """
    rate = (_p_y1) / (1.0 - _p_y1)
    for w in email:
        if is_use_not_appear == False and (w not in _p_xi_y1 or w not in _p_xi_y0):
            continue
        rate *= _p_xi_y1.get(w, _p_xi_y0_not_appear) / _p_xi_y0.get(w, _p_xi_y1_not_appear)
        
    return rate


# 随机选取测试用例
test_spam_count = 2000; test_ham_count = 2000
enron_test_ham_fns = dict(zip(random.sample(enron_hams.keys(), test_ham_count), [1] * 2000))
enron_test_spam_fns = dict(zip(random.sample(enron_spams.keys(), test_ham_count), [1] * 2000))

# 统计每个单词分别在正常邮件、垃圾邮件中出现的次数
enron_ham_word_count = word_count(enron_hams, enron_test_ham_fns)
enron_spam_word_count = word_count(enron_spams, enron_test_spam_fns)
print(len(enron_ham_word_count), len(enron_spam_word_count))

# 计算概率
p_xi_y0 = {}; p_xi_y1 ={}
for w in enron_ham_word_count:
    p_xi_y1[w] = (enron_ham_word_count[w] + 1.0) / (len(enron_hams) + 2.0)  # 单词xi在正常邮件中出现的概率
for w in enron_spam_word_count:
    p_xi_y0[w] = (enron_spam_word_count[w] + 1.0) / (len(enron_spams) + 2.0)  # 单词xi在垃圾邮件中出现的概率
p_y1 = float(len(enron_hams)) / float(len(enron_hams) + len(enron_spams))  # 正常邮件的概率
p_xi_y1_not_appear = 1.0 / (2.0 + len(enron_spams))  # 拉普拉斯平滑，给未见过的词一个很小的概率
p_xi_y0_not_appear = 1.0 / (2.0 + len(enron_hams))  # 拉普拉斯平滑，给未见过的词一个很小的概率
```

enron_email数据中取出的2000个测试集进行测试。可以看到成功率在98%以上。
``` python
# 测试
err_ham_count = 0; err_spam_count = 0
for fn in enron_test_ham_fns:
    rate = calc(enron_hams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear)
    if rate < 1:
        err_ham_count += 1

for fn in enron_test_spam_fns:
    rate = calc(enron_spams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear)
    if rate >= 1:
        err_spam_count += 1

print('error ham: %s/%s (%.2f%%); error spam: %s/%s (%.2f%%)' % \
      (err_ham_count, test_ham_count, 100.0 * err_ham_count / test_ham_count,\
       err_spam_count, test_spam_count, 100.0 * err_spam_count / test_spam_count))

# 忽略没有出现过的单词
err_ham_count = 0; err_spam_count = 0
for fn in enron_test_ham_fns:
    rate = calc(enron_hams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear, False)
    if rate < 1:
        err_ham_count += 1

for fn in enron_test_spam_fns:
    rate = calc(enron_spams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear, False)
    if rate >= 1:
        err_spam_count += 1

print('error ham: %s/%s (%.2f%%); error spam: %s/%s (%.2f%%)' % \
      (err_ham_count, test_ham_count, 100.0 * err_ham_count / test_ham_count,\
       err_spam_count, test_spam_count, 100.0 * err_spam_count / test_spam_count))
```
以上代码输出如下：
`error ham: 11/2000 (0.55%); error spam: 27/2000 (1.35%)`
`error ham: 12/2000 (0.60%); error spam: 53/2000 (2.65%)`

然后使用ling_email数据作为测试集来试试：
``` python
# 加载ling_email数据
ling_hams, ling_spams = load_data_set('ling_email')

# 测试ling_email数据
err_ling_ham_count = 0; err_ling_spam_count = 0
for fn in ling_hams:
    rate = calc(ling_hams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear)
    if rate < 1:
        err_ling_ham_count += 1

for fn in ling_spams:
    rate = calc(ling_spams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear)
    if rate >= 1:
        err_ling_spam_count += 1

print('error ham: %s/%s (%.2f%%); error spam: %s/%s (%.2f%%)' % \
      (err_ling_ham_count, len(ling_hams), 100.0 * err_ling_ham_count / len(ling_hams),\
       err_ling_spam_count, len(ling_spams), 100.0 * err_ling_spam_count / len(ling_spams)))

# 忽略没有出现过的单词
err_ling_ham_count = 0; err_ling_spam_count = 0
for fn in ling_hams:
    rate = calc(ling_hams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear, False)
    if rate < 1:
        err_ling_ham_count += 1

for fn in ling_spams:
    rate = calc(ling_spams[fn], p_y1, p_xi_y0, p_xi_y1, p_xi_y0_not_appear, p_xi_y1_not_appear, False)
    if rate >= 1:
        err_ling_spam_count += 1

print('error ham: %s/%s (%.2f%%); error spam: %s/%s (%.2f%%)' % \
      (err_ling_ham_count, len(ling_hams), 100.0 * err_ling_ham_count / len(ling_hams),\
       err_ling_spam_count, len(ling_spams), 100.0 * err_ling_spam_count / len(ling_spams)))
```
输出结果如下。
`error ham: 574/2410 (23.82%); error spam: 13/481 (2.70%)`
`error ham: 283/2410 (11.74%); error spam: 14/481 (2.91%)`

可以看到对垃圾邮件的识别还是可以的。但对正常邮件的识别效果不太好，如果考虑了没有出现过的单词，会将21%的正常邮件识别为垃圾邮件，如果不考虑没有出现过的单词，则会将11%的正常邮件识别为垃圾邮件。这里识别效果没那么好的原因可能是样本的数据集不够大，对测试集中单词的覆盖不完全。

## 参考资料
- 吴恩达老师机器学习视频
- 统计学习方法 李航
- 机器学习 周志华
- 机器学习实战