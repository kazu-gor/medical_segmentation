import numpy as np
import cv2
import glob
import os
import shutil
from sklearn.model_selection import train_test_split


###石灰化76か所を/nana_roku/にtrain val testをそれぞれ分ける


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


for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test7/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test7_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val_mask/*.png'):
    os.remove(file)

files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_masks/*.png")

while True:
    train0, test = train_test_split(files1, test_size=65)
    cnt1 = 0
    for i in test:
        img = imread(i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)
        cnt1 += nlabels1 - 1
    if cnt1 == 76:
        while True:
            train, val = train_test_split(train0, train_size=0.5)
            cnt2 = 0
            for j in train:
                img = imread(j)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)
                cnt2 += nlabels1 - 1
            if cnt2 == 76:
                while True:
                    test3, test7 = train_test_split(test, train_size=20)
                    cnt3 = 0
                    for k in test3:
                        img = imread(k)

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)
                        cnt3 += nlabels1 - 1

                    if cnt3 == 23:
                        for i in train:
                            img = imread(i)
                            imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train_mask",
                                                 os.path.basename(i)), img)
                            shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images",
                                                     os.path.splitext(os.path.basename(i))[0] + ".png")
                                        , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train",
                                                       os.path.splitext(os.path.basename(i))[0] + ".png"))
                        for i in val:
                            img = imread(i)
                            imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val_mask",
                                                 os.path.basename(i)), img)
                            shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images",
                                                     os.path.splitext(os.path.basename(i))[0] + ".png")
                                        , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val",
                                                       os.path.splitext(os.path.basename(i))[0] + ".png"))

                        for i in test:
                            img = imread(i)
                            imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test_mask",
                                                 os.path.basename(i)), img)
                            shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images",
                                                     os.path.splitext(os.path.basename(i))[0] + ".png")
                                        , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test",
                                                       os.path.splitext(os.path.basename(i))[0] + ".png"))
                        for i in test3:
                            img = imread(i)
                            imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3_mask",
                                                 os.path.basename(i)), img)
                            shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images",
                                                     os.path.splitext(os.path.basename(i))[0] + ".png")
                                        , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3",
                                                       os.path.splitext(os.path.basename(i))[0] + ".png"))
                        for i in test7:
                            img = imread(i)
                            imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test7_mask",
                                                 os.path.basename(i)), img)
                            shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images",
                                                     os.path.splitext(os.path.basename(i))[0] + ".png")
                                        , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test7",
                                                       os.path.splitext(os.path.basename(i))[0] + ".png"))
                        break
                break
        break
