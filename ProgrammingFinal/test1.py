import os
import xlwt
import xlrd
import re
import numpy as np

data = [[0 for i in range(2)] for j in range(200)]

with open("60.txt", 'r') as f:
    lines = f.readlines()[1:]
    cnt = 0
    for line in lines:
        a = line.split('\t')
        data[cnt] = a
        cnt += 1


# coding=gbk
f = open('60.txt', 'r', encoding='ANSI')  # 打开数据文本文档，注意编码格式的影响，这里用的是ANSI编码
wb = xlwt.Workbook(encoding='ANSI')  # 打开一个excel文件
ws1 = wb.add_sheet('first')  # 添加一个新表
row = 0  # 写入的起始行
col = 0  # 写入的起始列
k = 0
for lines in f:
    a = lines.split(' ')  # txt文件中每行的内容按‘ ’分割并存入数组中
    k += 1
    #rb = xlrd.open_workbook('C:\\Users\\DELL\\Desktop\\biao.xlsx')
    #ws1 = rb.get_ws1(0)
    for i in range(len(a)):
        ws1.write(row, col, a[i])  # 向Excel文件中写入每一项
        col += 1
    row += 1
    col = 0
wb.save("excel文件路径")


