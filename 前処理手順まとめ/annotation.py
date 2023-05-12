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


###################前処理を行うコード#############
##############カット画像とマスク画像からカット画像に石灰化領域を載せた画像を作成#############

for file in glob.glob(r"C:/Users/student/pythonProject6/output_green/*.png"):
    os.remove(file)  ###########pngファイルを削除

files = glob.glob(r"C:/Users/student/pythonProject6/input/image/*.jpg") #カット画像

for i in files:
    img = imread(i)
    basename = os.path.splitext(os.path.basename(i))[0]
    mask = imread(r"C:/Users/student/pythonProject6/input/mask/" + basename + ".png") #マスク画像
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width = mask.shape
    for h in range(height):
        for w in range(width):
            if mask[h][w] != 0:
                img[h][w] = [0, 255, 0]

    imwrite(os.path.join("C:/Users/student/pythonProject6/output_green/", basename + "_green" + ".png"), img)
