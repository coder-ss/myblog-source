title: "LeetCode: Reverse Bits"
date: 2015-04-29 00:16:02
categories: LeetCode
tags: [LeetCode, Algorithm]
description: LeetCode中Reverse Bits和Number of 1 Bits的解题纪录。
---

## 问题描述
> Reverse bits of a given 32 bits unsigned integer.

> For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000). 
> ------[Reverse Bits](https://leetcode.com/problems/reverse-bits/)

## 解题方法
利用位操作可以写出很简短的代码。下面的代码进行了两次移位运算，其实可以只算一次而保存起来，已加快运行速度。
```
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t rs = 0;

        for(int i = 0; i < 32; ++i) {
            if(n & 1 << i) {
                rs += 1 << (31 - i);
            }
        }

        return rs;
    }
};
```

## 扩展
另一个类似的问题

> Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

> For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
> ------[Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)

代码基本一样
```
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int rs = 0;
        for(int i = 0; i < 32; ++i) {
        	if((n & 1<<i) != 0)
        		++rs;
        }

        return rs;
    }
};
```



