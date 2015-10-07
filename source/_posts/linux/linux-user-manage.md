title: Linux中用户管理
date: 2014-11-21 09:06:31
categories: Linux
tags: [Linux]
description: 这篇文章是[Linux就是这个范儿]的读书笔记。记录了用户管理有关的文件和相关的一些操作，如useradd、usermod、userdel、passwd等操作。然后重点介绍了sudo命令的配置和注意事项。
---

这篇文章是[Linux就是这个范儿](http://book.douban.com/subject/25918029/)的读书笔记。

## 相关文件
- `/etc/passwd` 用户信息存放文件
> 存储格式：
用户名:密码:UID:GID:用户全名:home 目录:shell
- `/etc/shadow` 密码信息存放文件
- `/etc/group` 组信息存放文件
> 格式：
组名:用户组密码:GID:用户组内的用户名
- `/etc/gshadow` 组密码存放文件

## 管理用户和组
- `useradd` 添加用户（`adduser`在不同的发行版上可能有不同的定义）
- `usermod` 编辑
- `userdel` 删除用户（-r选项会把home目录一同删掉）
- `passwd` 添加/修改命令
- `groupadd`、`groupmod`、`groupdel`、`gpasswd`

## sudo命令
在`/etc/sudoers`中可以配置用户或组具有sudo特权。
```
# 使testuser用户具有sudo特权，并能执行任何命令
testuser ALL=(ALL)   ALL

# 使wheel组具有sudo特权，并能执行任何命令
%wheel   ALL=(ALL) ALL

# 使wheel组具有sudo特权，并能执行任何命令，而且不需要密码
%wheel   ALL=(ALL) NOPASSWD: ALL

# 使users组可以执行以下两个命令，而其他命令都拒绝。（配置中，命令要写完整路径，两个命令之间用逗号“,”分隔。）
# 可执行命令：$sudo mount /mnt/cdrom
# 可执行命令：$sudo unmount /mnt/cdrom
%users ALL=/sbin/mount /mnt/cdrom, /sbin/unmount /mnt/cdrom

# 使users组不能利用sudo特权执行adduser、useradd，但可以执行其他操作
%users ALL=(ALL) ALL, !/usr/sbin/adduser, !/usr/sbin/useradd
```

su命令可以临时切换成其他用户，包括root，使用时需要输入切换用户的密码。
`su -` 切换到root用户并将当前目录变更 root主目录
`su username -` 切换到username用户并将当前目录变更为username的主目录

**当具有sudo特权的用户执行`$sudo su`时，会绕过输入root密码直接切换成root用户，因此，配置sudo特权时要取消利用sudo执行su操作的权限！！！**