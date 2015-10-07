title: PHP学习笔记--数组及有关函数
date: 2014-11-06 11:14:38
categories: php
tags: [php, php基本函数]
description: 记录了php数组有关的操作和函数，包括数组操作符、遍历数组、数组排序有关函数、从文件读入数组、浏览数组元素、数组计数、提取数组元素为标量等。
---

文章内容总结于[php圣经](http://book.douban.com/subject/3549421/)和[php官方文档](http://php.net/manual/zh/)。


## 数组操作符



| 操作符    |     名称 |   示例   |   结果   |
| :-------- | :-------- | :------ | :------ |
| +    |   联合 |  $a+$b  |  $a和$b的联合，数组$b将被附加到$a中，但是任何关键字冲突的元素将不会被添加  |
| ==   | 等价 | $a==$b | 如果$a与$b包含相同的元素，返回true |
| ===  | 恒等 | $a===$b | 如果$a与$b包含相同顺序和类型的元素，返回true |
| != | 不等价 | $a!=$b | 如果$a和$b不包含相同的元素，返回true |
| <> | 不等价 | $a<>$b | 与!=相同
|!== | 不恒等 | $a!==$b | 如果$a和$b不包含相同顺序类型的元素，返回true |






## 遍历数组



**foreach**
``` php
forech ($arr as $kye => $value) {
    echo $key." - ".$value."<br />";
}
```


**each**
``` php
reset($arr); //使用each()函数时，数组将指向当前记录的元素，使用next()、prev()等函数会改变当前记录的位置
while ($element = each($arr)) {
    echo $element['key']." - ".$element['value']."<br />";
}
```


**list和each**
``` php
reset($arr); //使用each()函数时，数组将指向当前记录的元素，使用next()、prev()等函数会改变当前记录的位置
while (list($key, $value) = each($arr)) {
    echo "$key - $value<br />";
}
```




## 数组排序


### 基本排序函数`sort()`

``` php
/*
* 对数组进行升序排序
* @param array &$array 要排序的数组
* @param int $sort_flags 指定排序行为
*            SORT_REGULAR - 正常比较单元（不改变类型） 
*            SORT_NUMERIC - 单元被作为数字来比较 
*            SORT_STRING - 单元被作为字符串来比较 
*            SORT_LOCALE_STRING - 根据当前的区域（locale）设置来把单元当作字符串比较，可以用 setlocale() 来改变。  
*            SORT_NATURAL - 和 natsort() 类似对每个单元以"自然的顺序"对字符串进行排序。 PHP 5.4.0 中新增的。 
*            SORT_FLAG_CASE - 能够与 SORT_STRING 或 SORT_NATURAL 合并（OR 位运算），不区分大小写排序字符串。 

*/
bool sort ( array &$array [, int $sort_flags = SORT_REGULAR ] )
```


### 关联数组排序`asort()`和`ksort()`

``` php
/*
*  对数组进行排序并保持索引关系
* @param array &$array 要排序的数组
* @param int $sort_flags 指定排序行为,参见sort()函数
*/
bool asort ( array &$array [, int $sort_flags = SORT_REGULAR ] )

/*
*  对数组按照键名排序
* @param array &$array 要排序的数组
* @param int $sort_flags 指定排序行为,参见sort()函数
*/
bool ksort ( array &$array [, int $sort_flags = SORT_REGULAR ] )
```


例如，对于数组`$fruits = array("d" => "lemon", "a" => "orange", "b" => "banana", "c" => "apple");
`，利用sort()排序，print_r()输出结果为：

``` php
Array
(
    [0] => apple
    [1] => banana
    [2] => lemon
    [3] => orange
)
```

利用asort()排序，print_r()输出结果为：

``` php
Array
(
    [c] => apple
    [b] => banana
    [d] => lemon
    [a] => orange
)
```

利用ksort()排序，print_r()输出结果为：

``` php
Array
(
    [a] => orange
    [b] => banana
    [c] => apple
    [d] => lemon
)
```

### 反向（降序）排序 `rsort()`、`arsort()`、`krsort()`

`rsort()`、`arsort()`、`krsort()`分别是`sort()`、`asort()`、`ksort()`的降序版本



### 用户定义排序`usort()`

对多于一维的数组进行排序，或者不按字母和数字的顺序进行排序，要用到用户定义排序。

``` php
/*
* 使用用户自定义的比较函数对数组中的值进行排序
* @param array &$array 要排序的数组
* @param callable（回调类型） $cmp_function 在第一个参数小于，等于或大于第二个参数时，该比较函数必须相应地返回一个小于，等于或大于 0 的整数
*/
bool usort ( array &$array , callable $cmp_function )
```

多维数组使用usort()的例子：
``` php
<?php
function cmp($a, $b)
{
    return strcmp($a["fruit"], $b["fruit"]);
}

$fruits[0]["fruit"] = "lemons";
$fruits[1]["fruit"] = "apples";
$fruits[2]["fruit"] = "grapes";

usort($fruits, "cmp");

while (list($key, $value) = each($fruits)) {
    echo "$fruits[$key]: " . $value["fruit"] . "\n";
}
?> 

//输出结果
$fruits[0]: apples
$fruits[1]: grapes
$fruits[2]: lemons
```

其他版本：`uasort()`、`uksort()`

### 随机排序`shuffle()`

``` php
/*
* 将数组打乱
* @param array &$array 待操作的数组
* @return bool 成功返回TRUE，失败返回FALSE
*/
bool shuffle ( array &$array )
```

### 数组反向`array_reverse()`

``` php
/*
* 返回一个单元顺序相反的数组
* @param array $array 输入的数组
* @param bool $preserve_keys 如果设置为 TRUE 会保留数字的键。 非数字的键则不受这个设置的影响，总是会被保留
*/
array array_reverse ( array $array [, bool $preserve_keys = false ] )
```


## 从文件载入数组`file()`

``` php
/*
* 把整个文件读入一个数组中
* @param string $filename
* @param int $flag 可选，可以是以下一个或多个常量： 
*                  FILE_USE_INCLUDE_PATH 在 include_path 中查找文件。 
*                  FILE_IGNORE_NEW_LINES 在数组每个元素的末尾不要添加换行符  
*                  FILE_SKIP_EMPTY_LINES 跳过空行 
* @param resource $context
* @return array 返回值，文件中的每行是数组的一个元素
*/
array file ( string $filename [, int $flags = 0 [, resource $context ]] )
```


## 其他数组操作

### 在数组中浏览

每个数组都有一个内部指针指向数组中的当前元素，php提供了一些函数直接使用和操作这个指针。

- `mixed current ( array &$array )`：返回数组中的当前单元,初始化一个数组后返回的是第一个元素，如果内部指针越过了数组的末端，返回FALSE
- `array each ( array &$array )`：返回数组中当前的键／值对并将数组指针向前移动一步，如果内部指针越过了数组的末端，返回FALSE
- `mixed next ( array &$array )`：将数组中的内部指针向前移动一位。返回移动后的元素值，如果没有更多元素，返回FALSE
- `mixed reset ( array &$array )`：将数组的内部指针指向第一个单元。返回数组第一个单元的值，如果数组为空返回FALSE
- `mixed end ( array &$array )`：将数组的内部指针指向最后一个单元.返回最后一个元素的值，或者如果是空数组返回FALSE
- `mixed next ( array &$array )`：将数组中的内部指针向前移动一位。返回数组内部指针指向的下一个单元的值，或当没有更多单元时返回FALSE
- `mixed prev ( array &$array )`：将数组的内部指针倒回一位。返回数组内部指针指向的前一个单元的值，或当没有更多单元时返回FALSE。
- `pos()`：current() 的别名


### 对数组中每个元素执行操作`array_walk()`

``` php
/*
* 使用用户自定义函数对数组中的每个元素做回调处理
* @param array &$array 输入的数组
* @param callable $funcname 对每个元素要执行的函数
*                           典型情况下 funcname 接受两个参数
*                           array 参数的值作为第一个，键名作为第二个
* @param mixed $userdata 如果提供了可选参数 userdata，将被作为第三个参数传递给 callback funcname
*/
bool array_walk ( array &$array , callable $funcname [, mixed $userdata = NULL ] )
```

### 统计数组元素个数：`count()`、`sizeof()`、`array_count_values()`

``` php
/*
* 计算数组中的单元数目或对象中的属性个数
* @param mixed $var 数组或者对象
* @param int $mode 如果设为COUNT_RECURSIV（或1），count() 将递归地对数组计数(不能统计无限递归)
*/
int count ( mixed $var [, int $mode = COUNT_NORMAL ] )

sizeof — count() 的别名


/*
* 统计数组中所有的值出现的次数
* @param array $input 要统计的数组
* @return array 一个关联数组，用input数组中的值作为键名，该值在数组中出现的次数作为值
*/
array array_count_values ( array $input )

```

### 将数组转换成标量变量`extract()`

``` php
/*
* 从数组中将变量导入到当前的符号表
* @param array &$var_array 输入的关联数组
* @param int $extract_type 对待非法/数字和冲突的键名的方法，以下值之一：
*                          EXTR_OVERWRITE 覆盖
*                          EXTR_SKIP 不覆盖
*                          EXTR_PREFIX_SAME 冲突时在变量名前加上前缀 prefix。 
*                          EXTR_PREFIX_ALL 所有变量名加上前缀 prefix
*                          EXTR_PREFIX_INVALID 仅在非法／数字的变量名前加上前缀 prefix
*                          EXTR_IF_EXISTS 仅在当前符号表中已有同名变量时，覆盖它们的值。其它的都不处理
*                          EXTR_PREFIX_IF_EXISTS 仅在当前符号表中已有同名变量时，建立附加了前缀的变量名，其它的都不处理
*                          EXTR_REFS 将变量作为引用提取
* @param string $prefix 前缀，extract_type 的值是 EXTR_PREFIX_SAME，EXTR_PREFIX_ALL，EXTR_PREFIX_INVALID 或 EXTR_PREFIX_IF_EXISTS 时需要
*/
int extract ( array &$var_array [, int $extract_type = EXTR_OVERWRITE [, string $prefix = NULL ]] )
```
