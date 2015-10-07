title: PHP学习笔记--字符串及有关函数
date: 2014-11-08 11:19:23
categories: php
tags: [php, php基本函数]
description: 记录了php字符串有关的操作。包括字符串格式化、字符串连接和分割、字符串比较等常见操作
---

文章内容总结于[php圣经](http://book.douban.com/subject/3549421/)和[php官方文档](http://php.net/manual/zh/)。



## 格式化字符串


### 清理空白字符

`trim()`、 `ltrim()`、 `rtrim()`
``` php
/*
* 去除字符串首尾处的空白字符（或者其他字符）
* @param string $str 待处理的字符串
* @param string $charlist 可选参数，过滤字符也可由 charlist 参数指定
* @return string 过滤后的字符串
*/
string trim ( string $str [, string $charlist = " \t\n\r\0\x0B" ] )

string ltrim ( string $str [, string $charlist ] )

string rtrim ( string $str [, string $charlist ] )

chop() — rtrim() 的别名
```


### 格式化字符串以便显示

**HTML格式化**
`nl2br()`
``` php
/*
* 在字符串所有新行之前插入 HTML 换行标记
* @param string $string 输入字符串
* @param bool $is_xhtml 可选参数，是否使用 XHTML 兼容换行符
* @return string 返回调整后的字符串
*/
string nl2br ( string $string [, bool $is_xhtml = true ] )
```

**为打印输出格式化**
`echo`、`print`
``` php
/* 输出一个或多个字符串。echo是一个语言结构而不是一个函数！
* @param string $arg 要输出的参数 
*/
void echo ( string $arg1 [, string $... ] )

/* 输出一个或多个字符串。print是一个语言结构而不是一个函数！
* @param string $arg 要输出的参数 
* @return int 总是返回 1
*/
int print ( string $arg )
```

`printf()`、`sprintf()`
``` php
/*
* 依据 format 格式参数返回字符串
* @param string $format 格式描述信息，具体的格式请参考官方文档
* @param mixed $args
* @return string 返回格式化后的字符串
*/
string sprintf ( string $format [, mixed $args [, mixed $... ]] )

/* 依据 format 格式参数产生输出
* @param string $format 格式描述信息
* @param mixed $args
* @return int 返回输出字符串的长度
*/
int printf ( string $format [, mixed $args [, mixed $... ]] )

// vsprintf、vprintf是相应的接收数组参数的版本
string vsprintf ( string $format , array $args )
int vprintf ( string $format , array $args )
```


**大小写转换**
`strtoupper()`、`strtolower()`、`ucfirst()`、`ucwords()`

``` php
/*
* 将字符串转化为大写
* @param string $string 输入字符串
* @param string 转换后的大写字符串
*/
string strtoupper ( string $string )

/*
* 将字符串转化为小写
* @param string $str 输入字符串
* @param string 转换后的小写字符串
*/
string strtolower ( string $str )

/*
* 将字符串的首字母转换为大写
* @param string $str 输入字符串
* @param string 返回结果字符串
*/
string ucfirst ( string $str )

/*
* 将字符串中每个单词的首字母转换为大写
* @param string $str 输入字符串
* @param string 返回转换后的字符串
*/
string ucwords ( string $str )
```


### 特殊字符转义

有些特殊字符在某些情况下（如数据库查询）会有特殊含义，要在字符串中包含这些字符应该进行转义。

PHP中有一个配置指令` magic_quotes_gpc`来控制是否自动转义，如果php配置中启动了该配置项，就不用执行下面介绍的函数。可以通过`get_magic_quotes_gpc()`函数检查该配置项是否启动。

```php
/*
* 使用反斜线引用字符串
* @param string $str 要转义的字符
* @param string 返回转义后的字符
*/
string addslashes ( string $str )

/*
* 反引用一个引用字符串
* @param string $str 输入字符串
* @param string 返回一个去除转义反斜线后的字符串（\' 转换为 ' 等等）。双反斜线（\\）被转换为单个反斜线（\）
*/
string stripslashes ( string $str )
```

强烈建议使用 DBMS 指定的转义函数 （比如 MySQL 是  `mysqli_real_escape_string()`，PostgreSQL 是  `pg_escape_string()`），但是如果你使用的 DBMS 没有一个转义函数，就使用`addslashes()`。



## 连接和分割字符串

`explode()`、`implode()`、`join()`

``` php
/*
* 使用一个字符串分割另一个字符串
* @param string $delimiter 边界上的分隔字符
* @param string $string 输入的字符串
* @param int $limit 返回数组元素个数上限
* @return array 此函数返回由字符串组成的 array，每个元素都是 string 的一个子串，它们被字符串 delimiter 作为边界点分割出来
*               如果 delimiter 为空字符串（""），explode() 将返回 FALSE
*/
array explode ( string $delimiter , string $string [, int $limit ] )

/*
* 将一个一维数组的值转化为字符串
* @param string $glue 默认为空的字符串
* @param array $pieces 想要转换的数组
* @return string 返回一个字符串，其内容为由 glue 分割开的数组的值
*/
string implode ( string $glue , array $pieces )
string implode ( array $pieces )

join — 别名 implode()
```

`strtok()`

``` php
/* 标记分割字符串
* @param string $str 被分成若干子字符串的原始字符串
* @param string $token 分割 str 时使用的分界字符
* @return string 标记后的字符串
* 注意仅第一次调用 strtok 函数时使用 string 参数。
* 后来每次调用 strtok，都将只使用 token 参数，因为它会记住它在字符串 string 中的位置。
* 如果要重新开始分割一个新的字符串，你需要再次使用 string 来调用 strtok 函数，以便完成初始化工作。
* 注意可以在 token 参数中使用多个字符。字符串将被该参数中任何一个字符分割
*/
string strtok ( string $str , string $token )
string strtok ( string $token )

//strtik()范例
<?php
$string = "This is\tan example\nstring";
/* 使用制表符和换行符作为分界符 */
$tok = strtok($string, " \n\t");

while ($tok !== false) {
    echo "Word=$tok<br />";
    $tok = strtok(" \n\t");
}
?> 
```


`substr()`
``` php
/* 返回字符串的子串
* @param string $string 输入字符串
* @param int $start 起始位置，从0开始。负数代表倒数第几个字符开始
* @param int $length 子串的长,负数代表到倒数第几个
* @return string 返回提取的子字符串， 或者在失败时返回 FALSE
*/
string substr ( string $string , int $start [, int $length ] )
```


## 字符串的比较

`strcmp()`、`strcasecmp()`、`strnatcmp()`、`strnatcasecmp()`

``` php
/* 按字典顺序比较字符串，区分大小写
* @param string $str1
* @param string $str2
* @return int 如果 str1 小于 str2 返回 < 0； 
*             如果 str1 大于 str2 返回 > 0；
*             如果两者相等，返回 0
*/
int strcmp ( string $str1 , string $str2 )

//strcmp的不区分大小写版本
int strcasecmp ( string $str1 , string $str2 )

/* 按自然排序算法比较字符串，区分大小写
* @param string $str1
* @param string $str2
* @return int 如果 str1 小于 str2 返回 < 0； 
*             如果 str1 大于 str2 返回 > 0；
*             如果两者相等，返回 0
*/
int strnatcmp ( string $str1 , string $str2 )

//strnatcmp的不区分大小写版本
int strnatcasecmp ( string $str1 , string $str2 )
```

`strlen()`
``` php
/*
* 获取字符串长度
* @param string $string
* @return int
*/
int strlen ( string $string )
```

## 匹配与替换

### 字符串中查找字符串

`strstr()`、`strchr()`、`strrchr()`、`stristr()`

``` php
/*
* 查找字符串的首次出现
* @param string $haystack 输入字符串
* @param mixed $needle 要查找的子字符串。
*                      如果needle不是字符串将被转化为整型并且作为字符的序号来使用
* @param bool $before_needle 若为 TRUE，strstr() 将返回 needle 在 haystack 中的位置之前的部分
* @return string 返回字符串的一部分或者 FALSE（如果未发现 needle）
*/
string strstr ( string $haystack , mixed $needle [, bool $before_needle = false ] )

strchr — 别名 strstr()

//strstr()不区分大小写的版本
string stristr ( string $haystack , mixed $needle [, bool $before_needle = false ] )

/*
* 查找指定字符在字符串中的最后一次出现
* @param string $haystack 输入字符串
* @param mixed $needle
* @param bool $before_needle
* @return string 该函数返回 haystack 字符串中的一部分，这部分以 needle 的最后出现位置开始，直到 haystack 末尾。或者返回false
*/
string strrchr ( string $haystack , mixed $needle )
```

### 查找子字符串的位置

`strpos()`、`strrpod()`
``` php
/*
* 查找字符串首次出现的位置
* @param string $haystack 输入字符串
* @param mixed $needle 如果 needle 不是一个字符串，那么它将被转换为整型并被视为字符的顺序值
* @param int $offset 如果提供了此参数，搜索会从字符串该字符数的起始位置开始统计，不能为负
* @return mixed 字符串起始的位置(独立于 offset)。没有匹配返回FALSE
*/
mixed strpos ( string $haystack , mixed $needle [, int $offset = 0 ] )

/*
* 计算指定字符串在目标字符串中最后一次出现的位置
* @param string $haystack 输入字符串
* @param mixed $needle
* @param int $offset
* @return mixed
*/
int strrpos ( string $haystack , string $needle [, int $offset = 0 ] )

//对应不区分大小写的版本
int stripos ( string $haystack , string $needle [, int $offset = 0 ] )
int strripos ( string $haystack , string $needle [, int $offset = 0 ] )
```

- 注意判断返回第0个位置和返回false，使用运算符“===”
- 仅用于判断某个子串是否存在时，使用strpos比strstr速度快


### 替换子字符串

`str_replace()`、`substr_replace()`

``` php
/* 子字符串替换
* @param mixed $search 查找的目标值，也就是 needle。一个数组可以指定多个目标
* @param mixed $replace search 的替换值。一个数组可以被用来指定多重替换
* @param mixed $subject 执行替换的数组或者字符串
* @param int &$count 如果被指定，它的值将被设置为替换发生的次数
* @return mixed 该函数返回替换后的数组或者字符串
*/
mixed str_replace ( mixed $search , mixed $replace , mixed $subject [, int &$count ] )

/* 替换字符串的子串
* @param mixed $string 输入字符串或数组
* @param mixed $replacement 替换字符串
* @param mixed $start 起始位置
* @param mixed $length 被替换子串长度
* @return 返回结果字符串。如果 string 是个数组，那么也将返回一个数组
*/
mixed substr_replace ( mixed $string , mixed $replacement , mixed $start [, mixed $length ] )
```