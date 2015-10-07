title: "CentOS 6.5升级Python到2.7"
date: 2015-04-30 03:07:48
categories: Linux
tags: [Linux, Python]
description: CentOS 6.5默认的Python版本为2.6.6，官方yum源也没有提供2.7的版本。只能通过源码安装。
---

## 升级CentOS，安装开发者工具
```
yum -y update
yum groupinstall -y 'development tools'
```
如果在安装`development tools`的时候提示需要`kernel-devel`依赖项：
```
Error: Package: systemtap-devel-2.5-5.el6.x86_64 (base)            Requires: kernel-devel
```
则将`/etc/yum.conf`文件中的`exclude=kernel*`注释掉再安装`development tools`：
```
# PUT YOUR REPOS HERE OR IN separate files named file.repo
# in /etc/yum.repos.d
# exclude=kernel*
```
安装`SSL, bz2, zlib`等工具
```
yum install -y zlib-devel bzip2-devel openssl-devel xz-libs wget
```

## 源码安装Python2.7
```
wget https://www.python.org/ftp/python/2.7.9/Python-2.7.9.tgz
tar zxvf Python-2.7.9.tgz

cd Python-2.7.9.tgz
./configure
make
make altinstall
```
检查版本
`/usr/local/bin/python2.7 -V`

## 配置
```
mv /usr/bin/python /usr/bin/python2.6.6  
ln -s /usr/local/bin/python2.7 /usr/bin/python
```
检查版本
`python -V`

解决系统 Python 软链接指向 Python2.7 版本后，因为yum是不兼容 Python 2.7的，所以yum不能正常工作，我们需要指定 yum 的Python版本
```
vim /usr/bin/yum
```
将文件头部的`#!/usr/bin/python`改成`#!/usr/bin/python2.6.6`