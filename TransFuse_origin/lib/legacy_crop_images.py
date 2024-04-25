import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

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
###################↓3/4の画像を作る############
##################ガウシアンフィルタの挙動だけmatlabのコード違うがそれ以外はmatlabと同じ##################

## 前処理1のimage, maskからpngファイルを削除し，中身を掃除 ##
# for file in glob.glob(r"C:/Users/student/pythonProject6/前処理1/image/*.png"):
#     os.remove(file)
# for file in glob.glob(r"C:/Users/student/pythonProject6/前処理1/mask/*.png"):
#     os.remove(file)


def legacy_prerocessing():
    files = glob.glob('../../../dataset/original_images/images/*.jpg')
    print(f"Number of files: {len(files)}")

    os.makedirs('../../../dataset/original_images/preprocessed/images', exist_ok=True)
    os.makedirs('../../../dataset/original_images/preprocessed/masks', exist_ok=True)

    for i in tqdm(files):
        img = imread(i)
        basename = os.path.splitext(os.path.basename(i))[0]
        mask = imread(f'../../../dataset/original_images/masks/{basename}.png')
        # mask = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        img = img[math.floor(h * 3 / 4) - 1:, :]
        mask = mask[math.floor(h * 3 / 4) - 1:, :]  #############################
        h, w = img.shape
        img_w = []

        # ↓画像を横方向にスキャンし，横1行分の領域を5分割する．
        # 分割したそれぞれの領域内で輝度値の標準偏差を計算し，
        # 各領域の標準偏差がしきい値未満のものが3領域以上存在するとき，
        # その行を削除する．
        h_list = []
        for height in range(h):
            img_h = img[height, :]
            img_list = [img_h[:w // 5], img_h[w // 5:int(w / 5 * 2)], img_h[int(w / 5 * 2):int(w / 5 * 3)],
                        img_h[int(w / 5 * 3):int(w / 5 * 4)], img_h[int(w / 5 * 4):]]
            count = 0
            for img_w in img_list:
                if np.std(img_w) < 5:
                    count += 1
            if count >= 3:
                h_list.append(height)
        img = np.delete(img, h_list, 0)
        mask = np.delete(mask, h_list, 0)  ##############################

        # ↓スケール線の削除
        # 注目画素と左右の画素間の各々の輝度値の差が同時に50以上となった場合，
        # 注目画素をその左右の画素の平均値で置き換えることで，スケール線を削除する．
        # スケール線が2画素にまたがっている場合もあるため，注目画素から2画素離れた，左右の画素を用いて同様に処理を行う．
        # また，横1列にスケール線が存在する場合もある．
        # このような場合は，画像を縦方向に走査することでスケール線の特定を行う．
        # この場合は注目画素とその上下の画素を用いて削除を行う．
        height, width = img.shape
        for h in range(1, height - 1):
            for w in range(1, width - 1):
                if abs(int(img[h, w]) - int(img[h, w - 1])) >= 50 and abs(int(img[h, w]) - int(img[h, w + 1])) >= 50:
                    img[h, w] = ((img[h, w - 1].astype('uint16') + img[h, w + 1].astype('uint16')) // 2).astype('uint8')
                if abs(int(img[h, w]) - int(img[h - 1, w])) >= 50 and abs(int(img[h, w]) - int(img[h + 1, w])) >= 50:
                    img[h, w] = ((img[h - 1, w].astype('uint16') + img[h + 1, w].astype('uint16')) // 2).astype('uint8')
        for h in range(2, height - 2):
            for w in range(2, width - 2):
                if abs(int(img[h, w]) - int(img[h, w - 2])) >= 50 and abs(int(img[h, w]) - int(img[h, w + 2])) >= 50:
                    img[h, w] = ((img[h, w - 2].astype('uint16') + img[h, w + 2].astype('uint16')) // 2).astype('uint8')
                if abs(int(img[h, w]) - int(img[h - 2, w])) >= 50 and abs(int(img[h, w]) - int(img[h + 2, w])) >= 50:
                    img[h, w] = ((img[h - 2, w].astype('uint16') + img[h + 2, w].astype('uint16')) // 2).astype('uint8')

        img = cv2.GaussianBlur(img, (7, 7), 0.5)

        ## 前処理1のimage,maskフォルダに下部1/4の画像を出力 ##
        imwrite(os.path.join(f"../../../dataset/original_images/preprocessed/images/{basename}.png"), img)
        imwrite(os.path.join(f"../../../dataset/original_images/preprocessed/masks/{basename}.png"), mask)
