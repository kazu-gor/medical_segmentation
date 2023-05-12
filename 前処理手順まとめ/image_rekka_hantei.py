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


for file in glob.glob(r"C:/Users/student/PycharmProjects/lab/output/rekka_hantei/*.png"):
    os.remove(file)


files = glob.glob(r"C:/Users/student/PycharmProjects/lab/dataset/R_image/*.jpg")
for i in files:
    img = imread(i)
    basename = os.path.splitext(os.path.basename(i))[0]
    img2 = imread("C:/Users/student/PycharmProjects/lab/output/r_remove/" + basename+".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    output = np.uint8(np.zeros((height, width)))
    for h in range(height):
        for w in range(width):
            if img[h][w] != img2[h][w]:
                output[h][w] = 255

    imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/output/rekka_hantei", basename + ".png"), output)
