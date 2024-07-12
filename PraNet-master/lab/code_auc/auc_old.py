import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def all_area(files1, files2):  #############正解画像ではなく出力画像のフォルダ##########
    all_area_list = []
    for i in files1:
        basename = os.path.basename(i)
        img = imread(files2 + basename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)
        for j in range(1, nlabels):
            all_area_list.append(np.sum(labels == j))
    all_area_list.append(1)
    all_area_list.append(416 * 416)
    all_area_list = sorted(list(set(all_area_list)))
    return all_area_list


######################################################################
####################面積としきい値両方動かしてAUCを求める###############
def test(files1, files2, thresh, lower_size, upper_size=416 * 416 + 1):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in files1:
        basename = os.path.basename(i)
        img = imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)

        img2 = imread(files2 + basename)
        t2, img2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        nlabels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(gray2)

        top2_idx2 = []
        for j in range(1, nlabels2):
            area = np.sum(labels2 == j)
            if (area >= lower_size) and (area <= upper_size):
                top2_idx2.append(j)

        flag = False
        count = 0
        if nlabels1 > 1:
            for k in range(1, nlabels1):
                for l in top2_idx2:
                    l1 = labels1 == k
                    l2 = labels2 == l
                    a = l1 * l2
                    if np.any(a):
                        TP += 1
                        count = 1
                        flag = True
                        break
                if flag:
                    break
            if count == 0:
                FN += 1
        if nlabels1 == 1:
            if len(top2_idx2) != 0:
                FP += 1
            else:
                TN += 1
    return TP, FN, FP, TN


TP_list = []
FN_list = []
FP_list = []
TN_list = []
thresh_list = []
lower_list = []
upper_list = []
fpr_tpr_list = []
False_Positive_Rate_list = []
True_Positive_Rate_list = []

n = 0
MTP = 0
MFN = 0
MFP = 0
MTN = 0
MF = 0
ML = 0
MU = 0
thresh_max = 0

files1 = glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png"
)  ###正解画像
files2 = "/home/student/src2/藤林/プログラム/PraNet-master/results/PraNet/"
areas = all_area(files1, files2)
# for thresh in range(-1, 256):
for thresh in [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    128,
    130,
    140,
    150,
    160,
    170,
    180,
    190,
    200,
    210,
    220,
    230,
    240,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
]:
    for lower_size in areas:
        for upper_size in areas:
            if lower_size < upper_size:
                TP, FN, FP, TN = test(files1, files2, thresh, lower_size, upper_size)
                TP_list.append(TP)
                FN_list.append(FN)
                FP_list.append(FP)
                TN_list.append(TN)
                if (FP + TN) != 0 and (TP + FN) != 0:
                    False_Positive_Rate = FP / (FP + TN)
                    True_Positive_Rate = TP / (TP + FN)
                    False_Positive_Rate_list.append(False_Positive_Rate)
                    True_Positive_Rate_list.append(True_Positive_Rate)

                thresh_list.append(thresh)
                lower_list.append(lower_size)
                upper_list.append(upper_size)
                n += 1
for i in range(n):
    TP = TP_list[i]
    FN = FN_list[i]
    FP = FP_list[i]
    TN = TN_list[i]
    if TP != 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = (2 * precision * recall) / (precision + recall)
        if MF <= F_measure:
            MF = F_measure
            MTP = TP
            MFN = FN
            MFP = FP
            MTN = TN
            ML = lower_list[i]
            MU = upper_list[i]
            thresh_max = thresh_list[i]

print("-----------------------")
print("----------MAX----------")
print("threshhold:", thresh_max)
print("lower_size:", ML)
print("upper_size:", MU)
print("TP:", MTP)
print("FN:", MFN)
print("FP:", MFP)
print("TN:", MTN)
accuracy = (MTP + MTN) / (MTP + MFN + MFP + MTN)
print("Accuracy:", accuracy)

if (MTN + MFP) != 0 and MTP != 0:
    precision = MTP / (MTP + MFP)
    recall = MTP / (MTP + MFN)
    F_measure = (2 * precision * recall) / (precision + recall)
    specificity = MTN / (MTN + MFP)
    print("F-measure:", F_measure)
    print("precision:", precision)
    print("recall:", recall)
    print("specificity:", specificity)

x = list(zip(False_Positive_Rate_list, True_Positive_Rate_list))
x.sort()

False_Positive_Rate_list, True_Positive_Rate_list = zip(*x)
#####################同じ座標削除##########################
n1 = 0
match_list = []
for fpr, tpr in zip(False_Positive_Rate_list, True_Positive_Rate_list):
    if n1 != 0:
        if [fpr, tpr] == [
            False_Positive_Rate_list[n1 - 1],
            True_Positive_Rate_list[n1 - 1],
        ]:
            match_list.append(n1)
    n1 += 1
False_Positive_Rate_list = list(False_Positive_Rate_list)
True_Positive_Rate_list = list(True_Positive_Rate_list)
for i in sorted(match_list, reverse=True):
    del False_Positive_Rate_list[i]
    del True_Positive_Rate_list[i]

#######################################################################################
#################x軸(fpr)が同じ値で３つ以上存在するときy軸(tpr)が一番小さいのと大きいの以外削除
n2 = 0
match_list2 = []
for fpr, tpr in zip(False_Positive_Rate_list, True_Positive_Rate_list):
    if n2 != 0 and n2 != 1:
        if fpr == False_Positive_Rate_list[n2 - 2]:
            match_list2.append(n2 - 1)
    n2 += 1
for i in sorted(match_list2, reverse=True):
    del False_Positive_Rate_list[i]
    del True_Positive_Rate_list[i]
#########################################################################################

# for fpr, tpr in zip(False_Positive_Rate_list, True_Positive_Rate_list):
#     print(fpr, tpr)
n3 = 0
auc = 0
for fpr, tpr in zip(False_Positive_Rate_list, True_Positive_Rate_list):
    if n3 != 0:
        auc += (
            (fpr - False_Positive_Rate_list[n3 - 1])
            * (tpr + True_Positive_Rate_list[n3 - 1])
            / 2
        )
    n3 += 1

print("AUC:", auc)
fig = plt.figure()
plt.plot(
    False_Positive_Rate_list,
    True_Positive_Rate_list,
    label="ROC curve (area = %.3f)" % auc,
)
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.show()
fig.savefig("fig/img.png")
