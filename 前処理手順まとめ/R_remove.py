import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os
from PIL import Image
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


for file in glob.glob(r"C:/Users/student/PycharmProjects/lab/output/r_remove/*.png"):
    os.remove(file)

template = imread("R.jpg")
temp = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
h, w = temp.shape
files = glob.glob(r"C:/Users/student/PycharmProjects/lab/dataset/R_image/*.jpg")
for i in files:
    img = imread(i)
    basename = os.path.splitext(os.path.basename(i))[0]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h2, w2 = gray.shape

    match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt

    # テンプレートマッチングの結果を出力
    # cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    # print(f"{basename}:({pt[0]}, {pt[1]}),({pt[0] + w}, {pt[1] + h}), shape{gray.shape}")

    ########################################以下の①か②どちらか選択して使用####################################
    ###########################################①長方形のmaskを処理###############################################
    mask = np.uint8(np.zeros((h2, w2)))
    cv2.rectangle(mask,
                  pt1=(49, 1410),
                  pt2=(113, 1484),
                  color=(255, 255, 255),
                  thickness=-1)
    ###########################################②アノテーションした画像を用意してある場合，その部分を処理#################
    # if w2 == 2860:
    #     mask = imread("R_mask1.png")
    # elif w2 ==2836:
    #     mask = imread("R_mask2.png")
    #############################################################################################################


    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/output/r_remove", basename + ".png"), img)
