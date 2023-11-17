import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os


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


def all_area(files1, files2):  #############正解画像ではなく出力画像のフォルダ##########
    all_area_list = []
    for i in files1:
        basename = os.path.basename(i)
        img = imread(files2 + basename)
        t, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)#######################計算量削減、これを消したらすべての輝度値の面積を求めることになり時間がかかる

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)
        for j in range(1, nlabels):
            all_area_list.append(np.sum(labels == j))
    all_area_list.append(0)
    all_area_list.append(352 * 352)
    all_area_list = sorted(list(set(all_area_list)))
    return all_area_list


def val_test(files1, files2, lower_size, upper_size=352 * 352 + 1):
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
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        nlabels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(gray2)

        if lower_size == 0:
            if nlabels1 > 1:
                TP += 1
            else:
                FP += 1
        else:
            top2_idx2 = []
            for j in range(1, nlabels2):
                area = np.sum(labels2 == j)
                if (area >= lower_size) and (area <= upper_size):
                    top2_idx2.append(j)

            if len(top2_idx2) != 0:
                if nlabels1 > 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if nlabels1 > 1:
                    FN += 1
                else:
                    TN += 1

    return TP, FN, FP, TN


TP_list = []
FN_list = []
FP_list = []
TN_list = []
lower_list = []
upper_list = []
fpr_tpr_list = []

n = 0
MTP = 0
MFN = 0
MFP = 0
MTN = 0
MF = 0
ML_list = []

# files1 = glob.glob("./dataset/TestDataset/masks/*.png")  ###正解画像
files1 = glob.glob("./dataset/ValDataset/masks/*.png")  ###正解画像

files2 = "./results/Transfuse_S/"


areas = all_area(files1, files2)


for lower_size in areas:
    TP, FN, FP, TN = val_test(files1, files2, lower_size)
    TP_list.append(TP)
    FN_list.append(FN)
    FP_list.append(FP)
    TN_list.append(TN)

    lower_list.append(lower_size)
    n += 1

for i in range(n):
    TP = TP_list[i]
    FN = FN_list[i]
    FP = FP_list[i]
    TN = TN_list[i]
    if TP != 0:
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if MF < (TPR - FPR):
            MTP = TP
            MFN = FN
            MFP = FP
            MTN = TN
            MF = TPR - FPR
            ML_list = []
            ML_list.append(lower_list[i])
        elif MF == (TPR - FPR):
            MTP = TP
            MFN = FN
            MFP = FP
            MTN = TN
            ML_list.append(lower_list[i])

# ML = sum(ML_list) // len(ML_list)
i = len(ML_list)
ML = ML_list[i // 2]

print('------------------------------')
print('---------Youden index---------')
print("Threshold:", ML)
TPR = MTP / (MTP + MFN)
FPR = MFP / (MFP + MTN)
print("TPR-FPR:", TPR-FPR)
print("TP:", MTP)
print("FN:", MFN)
print("FP:", MFP)
print("TN:", MTN)
accuracy = (MTP + MTN) / (MTP + MFN + MFP + MTN)
precision = MTP / (MTP + MFP)
recall = MTP / (MTP + MFN)
F_measure = (2 * precision * recall) / (precision + recall)
specificity = MTN / (MTN + MFP)
print("Accuracy:", accuracy)
print("F-measure:", F_measure)
print("precision:", precision)
print("recall:", recall)
print("specificity:", specificity)
