import argparse
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import imageio
from torchvision import transforms

from lib.TransFuse_l_map import TransFuse_L

from lib.Discriminator_ResNet import Discriminator

from utils.dataloader import test_dataset
from skimage import img_as_ubyte

matplotlib.use("Agg")

#################################################
# epoch1~20の重みを全部テストする。

k = 5
thresh = 160
lower_size = 300
upper_size = 43264
total = 0
correct = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + \
        np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + \
        np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


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


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./snapshots/TransFuse_nash_mtl_b8/Transfuse-99.pth')
parser.add_argument('--pth_path2', type=str,
                    default='./snapshots/TransFuse_nash_mtl_b8/Discriminator-99.pth')
parser.add_argument('--save_path', type=str,
                    default='./results/Transfuse_S/', help='path to result')
parser.add_argument('--data_path1', type=str,
                    default='./dataset/TestDataset/', help='path to dataset')
parser.add_argument('--data_path2', type=str,
                    default='./dataset/sekkai_TestDataset/', help='path to only sekkai dataset')


opt = parser.parse_args()
save_path = opt.save_path
data_path1 = opt.data_path1
data_path2 = opt.data_path2

for arg_name, value in vars(opt).items():
    print(f'{arg_name}: {value}')

model = TransFuse_L()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

model2 = Discriminator()

model2.load_state_dict(torch.load(opt.pth_path2))

model2.cuda()
model2.eval()

image_root1 = '{}/images/'.format(data_path1)
gt_root1 = '{}/masks/'.format(data_path1)
test_loader1 = test_dataset(image_root1, gt_root1, opt.testsize)

image_root2 = '{}/images/'.format(data_path2)
gt_root2 = '{}/masks/'.format(data_path2)
test_loader2 = test_dataset(image_root2, gt_root2, opt.testsize)

dice_bank = []
iou_bank = []
acc_bank = []

y_true = np.array([])
y_score = np.array([])
y_pred = np.array([])

for i in range(test_loader1.size):
    image, gt, name = test_loader1.load_data()
    label = transforms.functional.to_tensor(gt)
    label = torch.einsum("ijk->i", label) > 0
    label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
    gt = np.asarray(gt, np.float32)

    # gt /= (gt.max() + 1e-8)  ##########################

    _image = image.mul(torch.FloatTensor(
        IMAGENET_STD).view(3, 1, 1))
    _image = _image.add(torch.FloatTensor(
        IMAGENET_MEAN).view(3, 1, 1)).detach()

    image = image.cuda()

    with torch.no_grad():
        _, _, res, att_mat = model(image)

        ###########  attention map  ############
        att_mat = torch.stack(att_mat).squeeze(1).detach().cpu()
        att_mat = torch.mean(att_mat, dim=1)

        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))

        # 入力から出力までのattention map
        # for i, v in enumerate(joint_attentions):
        # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = v[0, :].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), _image.shape[2:])[..., np.newaxis]
        # result = (mask * (_image.numpy().transpose(0, 2, 3, 1)[0] * 255)).astype(np.uint8)
        result = (mask * 255).astype(np.uint8)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map')
        _ = ax1.imshow(((_image.numpy().transpose(0, 2, 3, 1)[0]) * 255).astype(np.uint8))
        _ = ax2.imshow(result, cmap="hot")
        #########################################

        res1 = F.upsample(res, size=gt.shape,
                          mode='bilinear', align_corners=False)
        res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        res1 = 1. * (res1 > 0.5)

        imageio.imsave(save_path + name, img_as_ubyte(res1))
        res = res.repeat(1, 3, 1, 1)

        out = model2(res)

        _, predicted = torch.max(out, dim=1)
        predicted = torch.Tensor.cpu(predicted).detach().numpy()

        label = torch.Tensor.cpu(label).detach().numpy()
        out = torch.Tensor.cpu(out).detach().numpy()
        y_true = np.append(y_true, label)
        y_score = np.append(y_score, out[0][1] - out[0][0])
        y_pred = np.append(y_pred, predicted)

    if label == 1:
        if predicted == 1:
            imageio.imsave(save_path + 'TP/' + name, img_as_ubyte(res1))
            #########################################
            plt.savefig(f'./attention_map/TP/attention_map_{name}.png')
            #########################################
        else:
            imageio.imsave(save_path + 'FN/' + name, img_as_ubyte(res1))
            #########################################
            plt.savefig(f'./attention_map/FN/attention_map_{name}.png')
            #########################################

    else:
        if predicted == 1:
            imageio.imsave(save_path + 'FP/' + name, img_as_ubyte(res1))
            #########################################
            plt.savefig(f'./attention_map/FP/attention_map_{name}.png')
            #########################################

        else:
            imageio.imsave(save_path + 'TN/' + name, img_as_ubyte(res1))
            #########################################
            plt.savefig(f'./attention_map/TN/attention_map_{name}.png')
    plt.close()
    #########################################

# for i in range(test_loader2.size):
#     image, gt, name = test_loader2.load_data()
#     gt = np.asarray(gt, np.float32)

#     gt = 1. * (gt > 0.5)

#     image = image.cuda()

#     with torch.no_grad():
#         _, _, res = model(image)

#     res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#     res = res.sigmoid().data.cpu().numpy().squeeze()
#     # res = (res - res.min()) / (res.max() - res.min() + 1e-8)  ############################
#     res = 1. * (res > 0.5)

#     dice = mean_dice_np(gt, res)
#     iou = mean_iou_np(gt, res)
#     acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

#     acc_bank.append(acc)
#     dice_bank.append(dice)
#     iou_bank.append(iou)

# print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
#       format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.flatten()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print("----------discriminator-----------")
print("TPR-FPR:", TPR-FPR)
print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
print("TN:", TN)

if TP != 0:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F_measure = f1_score(y_true, y_pred)
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("F-measure:", F_measure)
    print("precision:", precision)
    print("recall:", recall)
    print("specificity:", specificity)

try:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fig = plt.figure()
    plt.ioff()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('./fig/roc_curve.png')

except Exception as e:
    print(e)

AUC = roc_auc_score(y_true, y_score)
print("AUC:", AUC)

print("-------------cutoff--------------")
Youden_index_candidates = tpr - fpr
index = np.where(Youden_index_candidates == max(Youden_index_candidates))[0][0]
cutoff = thresholds[index]
y_pred_cutoff = (y_score >= cutoff).astype(int)

cm = confusion_matrix(y_true, y_pred_cutoff)
TN, FP, FN, TP = cm.flatten()
print("TPR-FPR:", max(Youden_index_candidates))
print("Threshold:", cutoff)
print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
print("TN:", TN)

if TP != 0:
    accuracy = accuracy_score(y_true, y_pred_cutoff)
    precision = precision_score(y_true, y_pred_cutoff)
    recall = recall_score(y_true, y_pred_cutoff)
    F_measure = f1_score(y_true, y_pred_cutoff)
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("F-measure:", F_measure)
    print("precision:", precision)
    print("recall:", recall)
    print("specificity:", specificity)
