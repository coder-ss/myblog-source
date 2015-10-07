title: PHP学习笔记--文件及有关函数
date: 2014-11-11 13:18:09
categories: php
tags: [php, php基本函数]
description: 记录了php文件有关操作。包括文件读取、写入、打开、关闭，判断文件是否存在，返回文件大小，删除文件等操作
---

文章内容总结于[php圣经](http://book.douban.com/subject/3549421/)和[php官方文档](http://php.net/manual/zh/)。


## 打开、关闭文件

`fopen()`、`fclose()`
``` php
/*
* 打开文件或者 URL
* @param string $filename 文件路径或URL
* @param string $mode r、r+、w、w+、a、a+、x、x+、c、c+，组合使用：t、b
* @param bool $use_include_path 是否在 include_path 中搜寻文件
* @param resource $context
* @return resource 成功返回文件指针资源，失败返回false
*/
resource fopen ( string $filename , string $mode [, bool $use_include_path = false [, resource $context ]] )


/* 关闭一个已打开的文件指针
* @param resource $handle
* @return bool 成功返回TRUE，失败返回FALSE
*/
bool fclose ( resource $handle )
```


## 读文件

**每次读一行`fgets()`、`fgetss()`、`fgetcsv()`**
``` php
/*
* 从文件指针中读取一行
* @param resource $handle 文件指针
* @param int $length 从 handle 指向的文件中读取一行并返回长度最多为 length - 1 字节的字符串
* @return string 返回读取的字符串。没有更多的数据则返回 FALSE
*/
string fgets ( resource $handle [, int $length ] )


/*
* 从文件指针中读取一行并过滤掉 HTML 标记
* @param resource $handle 文件指针
* @param int $length 从 handle 指向的文件中读取一行并返回长度最多为 length - 1 字节的字符串
* @param string $allowable_tags 可选，指定哪些标记不被去掉
* @return string 返回读取的字符串。没有更多的数据则返回 FALSE
*/
string fgetss ( resource $handle [, int $length [, string $allowable_tags ]] )


/*
* 从文件指针中读取一行并过滤掉 HTML 标记
* @param resource $handle 文件指针
* @param int $length 必须大于 CVS 文件内最长的一行
* @param string $delimiter 设置字段分界符（只允许一个字符）
* @param string $enclosure 设置字段环绕符（只允许一个字符）
* @param string $escape 设置转义字符（只允许一个字符），默认是一个反斜杠
* @return array 返回包含读取字段的索引数组
*/
array fgetcsv ( resource $handle [, int $length = 0 [, string $delimiter = ',' [, string $enclosure = '"' [, string $escape = '\\' ]]]] )
```


**读取整个文件`readfile()`、`fpassthru()`、`file()`**

``` php
/*
* 读入一个文件并写入到输出缓冲，该函数不需要使用fopen打开文件
* @param string $filename 要读取的文件名
* @param  bool $use_include_path 是否在 include_path 中搜索文件
* @param resource $context
* @return int 返回从文件中读入的字节数，或者错误信息
*/
int readfile ( string $filename [, bool $use_include_path = false [, resource $context ]] )


/*
* 输出文件指针处的所有剩余数据
* @param resource $handle 文件指针
* @return int 返回从文件中读入的字节数，或者FALSE
*/
int fpassthru ( resource $handle )


/*
* 把整个文件读入一个数组中
* @param string $filename 文件的路径
* @param int $flags
* @param resource $context
* @return array 
*/
array file ( string $filename [, int $flags = 0 [, resource $context ]] )
```


**读取一个字符：`fgetc()`**
``` php
/*
* 把整个文件读入一个数组中
* @param resource $handle 文件指针
* @return string 返回一个包含有一个字符的字符串，碰到EOF返回FALSE
*/
string fgetc ( resource $handle )
```

**读取任意长度：`fread()`**
``` php
/*
* 从文件指针 handle 读取最多 length 个字节
* @param resource $handle 文件指针
* @param int $length 最多读取 length 个字节
* @return string 返回所读取的字符串，失败时返回 FALSE
*/
string fread ( resource $handle , int $length )
```


## 写文件
`fwrite()`、`fputs()`
``` php
/*
* 从文件指针 handle 读取最多 length 个字节
* @param resource $handle 文件指针
* @param string $string 要写入的字符串
* @param int $length 最多读取 length 个字节
* @return int 返回写入的字符数，错误则返回 FALSE 
*/
int fwrite ( resource $handle , string $string [, int $length ] )


fputs — fwrite() 的别名
```


## 其他函数

**查看文件是否存在`file_exists()`**
``` php
/*
* 检查文件或目录是否存在
* @param string $filename 
* @return bool 
*/
bool file_exists ( string $filename )
```

**确定文件大小`filesize()`**
``` php
/*
* 取得文件大小
* @param string $filename 
* @return int 返回文件大小字节数
*/
int filesize ( string $filename )
```

**删除一个文件`unlink()`**
``` php
/*
* 删除文件
* @param string $filename 文件的路径
* @param resource $context
* @return bool 成功TRUE，失败FALSE
*/
bool unlink ( string $filename [, resource $context ] )
```

**在文件中定位`rewind()`、`fseek()`、`ftell()`**
``` php
/*
* 将 handle 的文件位置指针设为文件流的开头
* @param resource $handle 文件指针
* @return bool 成功TRUE，失败FALSE
*/
bool rewind ( resource $handle )


/*
* 返回由 handle 指定的文件指针的位置，也就是文件流中的偏移量
* @param resource $handle 文件指针
* @return int 成功指针位置，失败FALSE
*/
int ftell ( resource $handle )


/*
* 设定文件指针位置
* @param resource $handle 文件指针
* @param int $offset 偏移量。$whence为SEEK_END时，$offset为负，表示从文件尾往前
* @param int $whence 为 SEEK_SET - 设定位置等于 offset 字节。
*                       SEEK_CUR - 设定位置为当前位置加上 offset。
*                       SEEK_END - 设定位置为文件尾加上 offset。
* @return int 成功则返回 0；否则返回 -1。注意移动到 EOF 之后的位置不算错误
*/
int fseek ( resource $handle , int $offset [, int $whence = SEEK_SET ] )
```


**文件锁定`flock()`**
``` php
/*
* 返回由 handle 指定的文件指针的位置，也就是文件流中的偏移量
* @param resource $handle 文件指针
* @param int $operation 为LOCK_SH取得共享锁定（读取的程序）。  
*                         LOCK_EX 取得独占锁定（写入的程序。  
*                         LOCK_UN 释放锁定（无论共享或独占）。
* @param int &$wouldblock
* @return int 成功返回 TRUE，失败返回 FALSE
*/
bool flock ( resource $handle , int $operation [, int &$wouldblock ] )
```