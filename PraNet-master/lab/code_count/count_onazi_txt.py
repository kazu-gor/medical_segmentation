import os
import sys

count = 0
with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/test3.txt", "r", encoding='utf-8') as f:
    datalist1 = f.readlines()
with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/train.txt", "r", encoding='utf-8') as f:
    datalist2 = f.readlines()
for data1 in datalist1:
    for data2 in datalist2:
        if "." in data1:
            data1 = os.path.splitext(os.path.basename(data1.rstrip('\n')))[0]
        else:
            data1 = data1.rstrip('\n')
        if "." in data2:
            data2 = os.path.splitext(os.path.basename(data2.rstrip('\n')))[0]
        else:
            data2 = data2.rstrip('\n')

        if data1 == data2:
            count += 1

print(count)
