import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.Trans_CaraNet import Trans_CaraNet_L
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import numpy as np
from adabelief_pytorch import AdaBelief


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(dataloaders_dict, model, optimizer, epoch, best_loss):
    val_loss = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        # ---- multi-scale training ----
        size_rates = [1]
        loss_record2, loss_record3, loss_record4, loss_record5= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                with torch.set_grad_enabled(phase == 'train'):
                    # ---- forward ----
                    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
                    # ---- loss function ----
                    loss5 = structure_loss(lateral_map_5, gts)
                    loss4 = structure_loss(lateral_map_4, gts)
                    loss3 = structure_loss(lateral_map_3, gts)
                    loss2 = structure_loss(lateral_map_2, gts)



                    loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
                    # loss = 0.5 * loss2 + 0.3 * loss3 + 0.15 * loss4 + 0.05 * loss5

                    # loss = 0.5 * (loss2 + loss3 + loss4 + loss5) + 0.5 * (
                    #         0.2 * loss_mapx + 0.3 * loss_map1 + 0.5 * loss_map2)
                    # ---- backward ----
                    if phase == 'train':
                        loss.backward()
                        # clip_gradient(optimizer, opt.clip)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
                        optimizer.step()

                        # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)


            if (i % 20 == 0 or i == total_step) and phase == 'train':
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
        if phase == 'train':
            train_loss = loss_record2.show() + loss_record3.show() + loss_record4.show() + loss_record5.show()
        elif phase == 'val':
            val_loss = loss_record2.show() + loss_record3.show() + loss_record4.show() + loss_record5.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), save_path + 'Transfuse-best.pth')
                print('[Saving best Snapshot:]', save_path + 'Transfuse-best.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'Transfuse-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'Transfuse-%d.pth' % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    return epoch, train_loss, val_loss, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    # parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    # parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    # parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='./dataset/ValDataset', help='path to val dataset')
    parser.add_argument('--train_path', type=str, default='./dataset/sekkai_TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='./dataset/sekkai_ValDataset', help='path to val dataset')
    parser.add_argument('--train_save', type=str, default='Transfuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device

    model = Trans_CaraNet_L(pretrained=True).cuda()

    # ---- flops and params ----

    params = model.parameters()

    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    image_root = Path(f'{opt.train_path}/images/')
    gt_root = Path(f'{opt.train_path}/masks/')

    image_root_val = Path(f'{opt.val_path}/images/')
    gt_root_val = Path(f'{opt.val_path}/masks/')

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    # train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, droplast=True)

    total_step = len(train_loader)

    val_loader = get_loader(image_root_val, gt_root_val, batchsize=opt.batchsize, trainsize=opt.trainsize, phase='val')

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    best_loss = 100000

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # train(train_loader, model, optimizer, epoch)

        epoch, train_loss, val_loss, best_loss = train(dataloaders_dict, model, optimizer, epoch, best_loss)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        epoch_list.append(epoch)

    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, label='train_loss')
    plt.plot(epoch_list, val_loss_list, label='val_loss', linestyle="--")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(left=0)
    plt.legend(loc='upper right')
    fig.savefig("fig/loss.png")
