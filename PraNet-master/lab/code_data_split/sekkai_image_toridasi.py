import numpy as np

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


##############石灰化の画像のみを取り出してsekkai_imagesとsekkai_maskに保存############

# for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_masks/*.png'):
#     os.remove(file)
# for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_images/*.png'):
#     os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TestDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TestDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TrainDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TrainDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_ValDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_ValDataset/images/*.png'):
    os.remove(file)

# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/masks/*.png")  ###正解画像
files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png")  ###正解画像

for i in files:
    basename = os.path.basename(i)
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(gray)
    nlabels = label[0]
    if nlabels > 1:
        # imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_masks",
        #                      os.path.basename(i)), img)
        # img2 = imread("/home/student/src2/藤林/プログラム/PraNet-master/dataset/images/" + basename)
        # imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_images",
        #                      os.path.basename(i)), img2)

        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TestDataset/masks",
                             os.path.basename(i)), img)
        img2 = imread("/home/student/src2/藤林/プログラム/PraNet-master/dataset/images/" + basename)
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TestDataset/images",
                             os.path.basename(i)), img2)

files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png")  ###正解画像

for i in files:
    basename = os.path.basename(i)
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(gray)
    nlabels = label[0]
    if nlabels > 1:
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TrainDataset/masks",
                             os.path.basename(i)), img)
        img2 = imread("/home/student/src2/藤林/プログラム/PraNet-master/dataset/images/" + basename)
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_TrainDataset/images",
                             os.path.basename(i)), img2)

files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png")  ###正解画像

for i in files:
    basename = os.path.basename(i)
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(gray)
    nlabels = label[0]
    if nlabels > 1:
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_ValDataset/masks",
                             os.path.basename(i)), img)
        img2 = imread("/home/student/src2/藤林/プログラム/PraNet-master/dataset/images/" + basename)
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/sekkai_ValDataset/images",
                             os.path.basename(i)), img2)