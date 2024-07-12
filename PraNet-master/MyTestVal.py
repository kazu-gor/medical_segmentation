import argparse
import glob
import os
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from lib.Cara.CaraNet import caranet
from lib.PraNet_Res2Net import PraNet as pranet
from lib.TransFuse_l import TransFuse_L
from lib.U_PraNet_Res2Net import U_PraNet as u_pranet
from scipy import misc
from skimage import img_as_ubyte
from utils.dataloader import test_dataset

##############################################
# テストデータと検証データを同時にテストして出力する。
# 検証データを使ってパラメータ調整するときに使う
# このあとkai_val_testでパラメータを調整


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = 0.001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


for file in glob.glob("./results/PraNet/*.png"):
    os.remove(file)

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
# parser.add_argument('--test_path', type=str, default='./dataset/TestDataset/', help='path to test dataset')
parser.add_argument("--normalization", type=bool, default=False)
# parser.add_argument('--model', type=str, default='pranet')
# parser.add_argument('--model', type=str, default='u_pranet')
parser.add_argument("--model", type=str, default="caranet")


# parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-99.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-59.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-90.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/PraNet/石灰化なし含む/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/U-PraNet/石灰化なし含む/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/CaraNet/石灰化なし含む/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/PraNet/石灰化ありのみ/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/U-PraNet/石灰化ありのみ/PraNet-best.pth')
parser.add_argument(
    "--pth_path",
    type=str,
    default="./weights/修論/segmentation/CaraNet/石灰化ありのみ/PraNet-best.pth",
)
# parser.add_argument('--pth_path', type=str, default='./weights/修論/discriminator_nash/PraNet_discriminator/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/discriminator_nash/U_PraNet_discriminator/PraNet-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/discriminator_nash/CaraNet_discriminator/PraNet-best.pth')


opt = parser.parse_args()

# data_path_list = [opt.test_path]
# data_path_list = ['./dataset/TestDataset/', './dataset/ValDataset/']
# data_path_list = ['./dataset/TestDataset/']
# data_path_list = ['./dataset/ValDataset/']
data_path_list = ["./dataset/sekkai_TestDataset/"]
# data_path_list = ['./dataset/sekkai_ValDataset/']

norm = opt.normalization
# norm = True

for data_path in data_path_list:
    save_path = "./results/PraNet/"

    if opt.model == "pranet" or opt.model == "p":
        model = pranet()
        print("model:pranet")
    elif opt.model == "u_pranet" or opt.model == "u":
        model = u_pranet()
        print("model:u_pranet")
    elif opt.model == "caranet" or opt.model == "c":
        model = caranet()
        print("model:caranet")

    model.load_state_dict(torch.load(opt.pth_path))
    # model.load_state_dict(torch.load(opt.pth_path), strict=False)

    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = "{}/images/".format(data_path)
    gt_root = "{}/masks/".format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    dice_bank = []
    iou_bank = []
    acc_bank = []

    no = 0
    time_start = time.time()
    for i in range(test_loader.size):
        no += 1
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)

        if norm:
            gt /= gt.max() + 1e-8  ##########################
        else:
            gt = 1.0 * (gt > 0.5)  ########################

        image = image.cuda()
        with torch.no_grad():
            res5, res4, res3, res2 = model(image)
        res = res2
        # res = res5
        res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        if norm:
            res = (res - res.min()) / (
                res.max() - res.min() + 1e-8
            )  ############################
        else:
            res = 1.0 * (res > 0.5)  ############################

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)
        imageio.imsave(save_path + name, img_as_ubyte(res))

    time_finish = time.time()
    print("timer: {:.4f} sec.".format((time_finish - time_start) / no))
    print(
        "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
            np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
        )
    )
