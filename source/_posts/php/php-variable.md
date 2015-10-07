title: PHP学习笔记--变量类型及有关函数
date: 2014-11-03 21:29:24
categories: php
tags: [php, php基本函数]
---

最近尝试着参加了3场面试，发现面试官对php基础知识的掌握还是很看中的。那么，php有关的一些基本函数都要做到心中有数。

今天要总结的是操作和测试变量的函数。知识内容来自[php圣经](http://book.douban.com/subject/3549421/)

## 测试和设置变量类型
```
string gettype(mixed var); //返回 PHP 变量的类型
bool settype(mixed var, string type); //将变量 var 的类型设置成 type
```
**gettype**可能的返回值有："boolean"、"integer"、 "double"、 "string"、 "array" 、"object"、 "resource"、"NULL"、"unknown type" 等。

> 不要使用 gettype() 来测试某种类型，因为其返回的字符串在未来的版本中可能需要改变。此外，由于包含了字符串的比较，它的运行也是较慢的。 
> 
使用 is_* 函数代替


**settype**中type 的可能值为： "boolean"（或为"bool"，从 PHP 4.2.0 起）、"integer"（或为"int"，从 PHP 4.2.0 起、"float" （只在 PHP 4.2.0 之后可以使用，对于旧版本中使用的"double"现已停用）、"string"、"array"、"object"、"null" （从 PHP 4.2.0 起） 

PHP推荐使用一些**特定类型的测试函数**：
- is_array()：检查变量是否是数组
- is_double()、is_float()、is_real()（所有都是相同的函数）：检查变量是否是浮点数
- is_long()、is_int()、is_integer()（所有都是相同的函数）：检查变量是否是整数
- is_string()：检查变量是否是字符串
- is_bool()：检查变量是否是布尔值
- is_object()：检查变量是否是一个对象
- is_resource()：检查变量是否是一个资源
- is_null()：检查变量是否是为null
- is_scalar()：检查该变量是否是标量，即，一个整数、布尔值、字符串或浮点数
- is_numeric()：检查该变量是否是任何类型的数字或数字字符串
- is_callable()：检查该变量是否是有效的函数名称



## 测试变量状态

**isset — 检测变量是否设置**
`
bool isset ( mixed $var [, mixed $... ] )
`


**unset — 释放给定的变量**
`
void unset ( mixed $var [, mixed $... ] )
`
unset() 在函数中的行为会依赖于想要销毁的变量的类型而有所不同。 
- 如果在函数中 unset() 一个全局变量，则只是局部变量被销毁，而在调用环境中的变量将保持调用 unset() 之前一样的值
- 如果您想在函数中 unset() 一个全局变量，可使用 $GLOBALS 数组来实现。
- 如果在函数中 unset() 一个通过引用传递的变量，则只是局部变量被销毁，而在调用环境中的变量将保持调用 unset() 之前一样的值。 
- 如果在函数中 unset() 一个静态变量，那么在函数内部此静态变量将被销毁。但是，当再次调用此函数时，此静态变量将被复原为上次被销毁之前的值。 


**empty — 检查一个变量是否为空**
`
bool empty ( mixed $var )
`
如果 var 是非空或非零的值，则 empty() 返回 FALSE。换句话说，""、0、"0"、NULL、FALSE、array()、var $var; 以及没有任何属性的对象都将被认为是空的，如果 var 为空，则返回 TRUE。 
> empty() 与 isset() 的一个简单比较。 
> ```
<?php
$var = 0;
> 
// 结果为 true，因为 $var 为空
if (empty($var)) {  
    echo '$var is either 0 or not set at all';
}
> 
// 结果为 false，因为 $var 已设置
if (!isset($var)) { 
    echo '$var is not set at all';
}
?> 
```


## 变量的重解释

**intval — 获取变量的整数值**
`
int intval ( mixed $var [, int $base = 10 ] )
`

**floatval — 获取变量的浮点值**
`
float floatval ( mixed $var )
`

**strval — 获取变量的字符串值**
`
string strval ( mixed $var )
`