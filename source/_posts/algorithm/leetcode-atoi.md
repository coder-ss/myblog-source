title: 【LeetCode】String to Integer (atoi)
date: 2014-11-30 20:58:25
categories: LeetCode
tags: [LeetCode, Algorithm]
description: LeetCode中String to Interger的解题纪录。
---

String to Interger是一个常用函数，但是要写出一个基本可用的其实不简单。在LeetCode上也遇到了这个[题目](https://oj.leetcode.com/problems/string-to-integer-atoi/)。

## 问题描述
> Implement atoi to convert a string to an integer.

## 解题方法

需要注意的各种情况
- 字符串前面的空白字符
- "+"、"-"符号
- 溢出
- 全空白，返回0
- 出现数字之前出现了非法字符，返回0
- 出现数字之后出现了非法字符，返回已求得的值

一段Accept的代码
```
class Solution {
public:
    int atoi(const char *str) {
        if(str == NULL)
            return 0;

        int pos = 0, rs = 0, multi = 1;

        unsigned int a = 0;
        int int_max = (~a) / 2;
        int int_min = -int_max - 1;

        while(str[pos] != '\0') {
            if(str[pos] == ' ' || str[pos] == '\t')
                ++pos;
            else
                break;
        }
        if(str[pos] == '-') {
            multi = -1;
            ++pos;
        } else if(str[pos] == '+') {
            multi = 1;
            ++pos;
        }
        bool ishavenum = false;
        while(str[pos] != '\0') {
            if(str[pos] >= '0' && str[pos] <= '9') {
                if(!ishavenum)
                    ishavenum = true;

                if(rs > int_max / 10)
                    if(multi == 1)
                        return int_max;
                    else
                        return int_min;
                if(rs == int_max / 10)
                    if(multi == 1 && (str[pos] - '0') > int_max % 10)
                        return int_max;
                    else if(multi == -1 && (str[pos] - '0') > (int_max % 10 + 1))
                        return int_min;

                rs = rs * 10 + (str[pos] - '0');
                ++pos;
            }else{
                if(ishavenum)
                    return rs * multi;
                else
                    return 0;
            }
        }

        return rs * multi;
    }
};
```