import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

import math


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


###################easyとnormalの正解画像から元の画像と同じサイズの白黒のマスク画像を生成#############


for file in glob.glob(r"C:/Users/student/pythonProject6/output/*.png"):
    os.remove(file) ##pngファイルを掃除##

files = glob.glob(r"C:/Users/student/pythonProject6/easy_normal_syusei/mask/*")

for file in files:
    mask = imread(file)
    basename = os.path.splitext(os.path.basename(file))[0]
    if 'green' in basename:
        basename = basename.replace('_green', '')
    img = imread(r"C:/Users/student/pythonProject6/easy_normal_syusei/image/" + basename + ".jpg")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    h2, w2, c2 = mask.shape

    if h != h2:
        mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(img, mask2, cv2.TM_SQDIFF)
        min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
        pt = min_pt


        ###########テンプレートマッチングした場所確認する画像を生成##########################
        # cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w2, pt[1] + h2), (0, 0, 200), 3)
        ###(x座標,y座標)#####
        # print(f"{basename}:({pt[0]}, {pt[1]}),({pt[0] + w2}, {pt[1] + h2}), shape{img.shape}")

        # imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/output/green_label_to_mask", basename + ".png"),
        #         img)
        ################################################################################

        ###############mask画像の緑色部分を白，それ以外を黒色にする#####################
        for i in range(h2):
            for j in range(w2):
                if mask[i][j][0] < 100 and mask[i][j][1] > 150 and mask[i][j][2] < 100:  #########BGR#######
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #############################################################################

        mask = cv2.copyMakeBorder(mask, pt[1], h - (pt[1] + h2), 0, 0, cv2.BORDER_CONSTANT)
        h3, w3 = mask.shape
        print(h3, w3, h, w, basename)

        if h3 != h or w3 != w:
            print('×', basename)

    ##outputフォルダに出力##
    imwrite(os.path.join("C:/Users/student/pythonProject6/output/", basename + ".png"), mask)
