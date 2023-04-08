import numpy as np

import cv2
import glob
import os


##予測結果のマスク画像からbounding_boxの座標を出して正解画像のマスク画像と重なっているか判定する


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


k = 5
lower_size = 200
upper_size = 100000
thresh = 200

TP = 0
FN = 0
FP = 0
files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png")  ###正解画像
for i in files:
    basename = os.path.basename(i)
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(gray)

    stats = label[2]

    data = label[2]

    bbox1 = []
    for i in range(1, label[0]):
        x0 = max(data[i][0] - k, 0)
        y0 = max(data[i][1] - k, 0)
        x1 = min(data[i][0] + data[i][2] + k, 415)
        y1 = min(data[i][1] + data[i][3] + k, 415)
        bbox1.append([x0, y0, x1, y1])

    img2 = imread("/home/student/src2/藤林/プログラム/PraNet-master/results/PraNet/" + basename)
    t2, img2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    label2 = cv2.connectedComponentsWithStats(gray2)

    stats2 = label2[2]

    area2 = stats2[:, cv2.CC_STAT_WIDTH] * stats2[:, cv2.CC_STAT_HEIGHT]
    area2 = area2[1:]
    n = 1
    top2_idx2 = []
    for i in area2:
        if (i > lower_size) and (i < upper_size):
            top2_idx2.append(n)
        n += 1

    data2 = label2[2]
    bbox2 = []
    for i in top2_idx2:
        x0 = max(data2[i][0] - k, 0)
        y0 = max(data2[i][1] - k, 0)
        x1 = min(data2[i][0] + data2[i][2] + k, 415)
        y1 = min(data2[i][1] + data2[i][3] + k, 415)
        bbox2.append([x0, y0, x1, y1])

    for i in bbox1:
        k = 0
        for j in bbox2:
            if (max(i[0], j[0]) <= min(i[2], j[2])) and (max(i[1], j[1]) <= min(i[3], j[3])):
                TP += 1
                k += 1
                break
        if k == 0:
            FN += 1
    for i in bbox2:
        k = 0
        for j in bbox1:
            if (max(i[0], j[0]) <= min(i[2], j[2])) and (max(i[1], j[1]) <= min(i[3], j[3])):
                k += 1
                break
        if k == 0:
            FP += 1

print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_measure = (2 * precision * recall) / (precision + recall)
print("F-measure:", F_measure)
print("precision:", precision)
print("recall:", recall)
