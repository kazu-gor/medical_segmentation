import argparse
import os
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Discriminator_ResNet import Discriminator
# from lib.TransFuse_l import TransFuse_L
from lib.pvt import PolypPVT
from torch.autograd import Variable
from utils.dataloader import get_loader
from utils.mtl import extract_weight_method_parameters_from_args
from utils.utils import AvgMeter, adjust_lr, clip_gradient
from utils.weight_methods import WeightMethods

matplotlib.use('Agg')
from utils.smooth_cross_entropy import SmoothCrossEntropy


def nash_analytic_v1(losses, shared_parameters, task_specific_parameters):
    extra_outputs = dict()

    grads = {}
    for i, loss in enumerate(losses):
        g = list(
            torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True
            )
        )

        grad = torch.cat([torch.flatten(grad) for grad in g])

        grads[i] = grad

    g1_g1 = torch.dot(grads[0].t(), grads[0])
    g2_g2 = torch.dot(grads[1].t(), grads[1])
    g1_g2 = torch.dot(grads[0].t(), grads[1])
    g2_g1 = torch.dot(grads[1].t(), grads[0])

    alpha_1 = torch.sqrt(
        torch.sqrt(g2_g2) / (g1_g1 * torch.sqrt(g2_g2) +
                             g1_g2 * torch.sqrt(g1_g1))
    )
    alpha_2 = torch.sqrt(
        torch.sqrt(g1_g1) / (g2_g1 * torch.sqrt(g2_g2) +
                             g2_g2 * torch.sqrt(g1_g1))
    )
    alpha = torch.tensor([alpha_1, alpha_2])

    weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
    extra_outputs["weights"] = alpha
    return weighted_loss, extra_outputs


def nash_analytic_v2(losses, shared_parameters, task_specific_parameters):
    extra_outputs = dict()

    grads = {}
    for i, loss in enumerate(losses):
        g = list(
            torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True
            )
        )

        grad = torch.cat([torch.flatten(grad) for grad in g])

        grads[i] = grad

    g1_g1 = torch.dot(grads[0].t(), grads[0])
    g2_g2 = torch.dot(grads[1].t(), grads[1])
    g1_g2 = torch.dot(grads[0].t(), grads[1])
    # g2_g1 = torch.dot(grads[1].t(), grads[0])

    cos_theta_g1g2 = g1_g2 / (torch.sqrt(g1_g1) * torch.sqrt(g2_g2))
    # cos_theta_g2g1 = g2_g1 / (torch.sqrt(g1_g1) * torch.sqrt(g2_g2))
    # assert cos_theta_g1g2 == cos_theta_g2g1, f"{cos_theta_g1g2} != {cos_theta_g2g1}"

    alpha_1 = 1 / (torch.sqrt(g1_g1 * (1 + cos_theta_g1g2)))
    alpha_2 = 1 / (torch.sqrt(g2_g2 * (1 + cos_theta_g1g2)))

    alpha = torch.tensor([alpha_1, alpha_2])

    weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
    extra_outputs["weights"] = alpha
    return weighted_loss, extra_outputs


def nash_analytic_v3(losses, shared_parameters, task_specific_parameters):
    extra_outputs = dict()

    grads = {}
    for i, loss in enumerate(losses):
        g = list(
            torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True
            )
        )

        grad = torch.cat([torch.flatten(grad) for grad in g])

        grads[i] = grad

    g1_g1 = torch.dot(grads[0].t(), grads[0])
    g2_g2 = torch.dot(grads[1].t(), grads[1])

    alpha_1 = 1 / torch.sqrt(g1_g1)
    alpha_2 = 1 / torch.sqrt(g2_g2)

    alpha = torch.tensor([alpha_1, alpha_2])

    weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
    extra_outputs["weights"] = alpha
    return weighted_loss, extra_outputs


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(dataloaders_dict, models, optimizer, criterion, epoch, best_loss, best2_loss):
    val_loss = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            for model in models.values():
                model.train()
        else:
            for model in models.values():
                model.eval()

        # ---- multi-scale training ----
        size_rates = [1]
        loss_res, loss_res1, d_loss_record = AvgMeter(), AvgMeter(), AvgMeter()

        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # d_optimizer.zero_grad()
                images, gts = pack
                labels = torch.einsum("ijkl->i", gts) > 0

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))
                # labels = labels.view(-1, 1)
                # labels = F.one_hot(labels, num_classes=2)

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                labels = Variable(labels).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                with torch.set_grad_enabled(phase == 'train'):
                    # ---- forward ----
                    res, res1 = models['Transfuse'](images)

                    # ---- loss function ----
                    l_res = structure_loss(res, gts)
                    l_res1 = structure_loss(res1, gts)
                    # d_loss = criterion(d_out, labels)
                    loss = 0.5 * l_res + 0.5 * l_res1
                    # TODO: try different weights for loss
                    # loss = loss1 + loss2 + loss3 + loss4 + loss5

                    # lateral_map_2 = lateral_map_2.sigmoid()#########################
                    # lateral_map_2 = 1. * (lateral_map_2 > 0.5)
                    # lateral_map_2 = images * lateral_map_2
                    res = (res + res1).repeat(1, 3, 1, 1)

                    d_out = models['Discriminator'](res)
                    # d_out = models['Discriminator'](lateral_map_2, images)
                    d_loss = criterion(d_out, labels)
                    # losses = [loss, d_loss]
                    losses = torch.stack((loss, d_loss))
                    # ---- backward ----
                    if phase == 'train':
                        if opt.mtl == 'nashmtl' and opt.analytic:
                            if opt.analytic_version == 'v1':
                                loss, _ = nash_analytic_v1(
                                    losses=losses,
                                    shared_parameters=shared_parameters,
                                    task_specific_parameters=task_specific_parameters,
                                )
                            elif opt.analytic_version == 'v2':
                                loss, _ = nash_analytic_v2(
                                    losses=losses,
                                    shared_parameters=shared_parameters,
                                    task_specific_parameters=task_specific_parameters,
                                )
                            elif opt.analytic_version == 'v3':
                                loss, _ = nash_analytic_v3(
                                    losses=losses,
                                    shared_parameters=shared_parameters,
                                    task_specific_parameters=task_specific_parameters,
                                )
                            else:
                                raise ValueError('Invalid value for analytic_version.')
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(shared_parameters, 1.0)
                        else:
                            loss, _ = weight_method.backward(
                                losses=losses,
                                shared_parameters=shared_parameters,
                                task_specific_parameters=task_specific_parameters,
                                # last_shared_parameters=list(model.last_shared_parameters()),
                                # representation=features,
                            )
                        # optimizer.pc_backward(losses)
                        clip_gradient(optimizer, opt.clip)
                        optimizer.step()

                        # loss.backward(retain_graph=True)
                        # clip_gradient(optimizer, opt.clip)
                        # optimizer.step()
                        # # optimizer.zero_grad()
                        # d_loss.backward()
                        # clip_gradient(optimizer, opt.clip)
                        # clip_gradient(d_optimizer, opt.clip)
                        # optimizer.step()
                        # d_optimizer.step()

                        # ---- recording loss ----
                if rate == 1:
                    d_loss_record.update(d_loss.data, opt.batchsize)
                    loss_res.update(loss.data, opt.batchsize)
                    loss_res1.update(loss.data, opt.batchsize)
            if (i % 20 == 0 or i == total_step) and phase == 'train':
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[loss_res: {:0.4f}, loss_res1: {:0.4f}, d_loss: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_res.show(), loss_res1.show(), d_loss_record.show()))
        if phase == 'train':
            train_loss = loss_res.show() + loss_res1.show()
            train_d_loss = d_loss_record.show()
        elif phase == 'val':
            val_loss = loss_res.show() + loss_res1.show()
            val_d_loss = d_loss_record.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best.pth')
                print('[Saving best Snapshot:]', save_path + 'TransFuse-best.pth')
            if val_d_loss < best2_loss:
                best2_loss = val_d_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best2.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best2.pth')
                print('[Saving best Snapshot:]', save_path + 'Discriminator-best2.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0:
        torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-%d.pth' % epoch)
        torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-%d.pth' % epoch)

        print('[Saving Snapshot:]', save_path + 'Transfuse-%d.pth' % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))

    return epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='./dataset/ValDataset', help='path to val dataset')
    parser.add_argument('--train_save', type=str, default='PolypPVT')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    parser.add_argument('--tuning', type=bool, default=True)

    # parser.add_argument('--mtl', type=str, default='nashmtl')
    # parser.add_argument('--mtl', type=str, default='pcgrad')
    # parser.add_argument('--mtl', type=str, default='cagrad')
    # parser.add_argument('--mtl', type=str, default='imtl')
    # parser.add_argument('--mtl', type=str, default='mgda')
    # parser.add_argument('--mtl', type=str, default='dwa')
    # parser.add_argument('--mtl', type=str, default='uw')
    # parser.add_argument('--mtl', type=str, default='ls')
    # parser.add_argument('--mtl', type=str, default='scaleinvls')
    # parser.add_argument('--mtl', type=str, default='rlw')
    parser.add_argument('--mtl', type=str, default='stl')

    parser.add_argument('--analytic', type=bool, default=False)
    parser.add_argument('--analytic_version', type=str, default='v1')

    opt = parser.parse_args()
    os.makedirs('./config', exist_ok=True)
    with open(f'./config/{opt.train_save}', 'w') as f:
        for arg_name, value in vars(opt).items():
            print(f'{arg_name}: {value}')
            f.write(f'{arg_name}: {value}')

    # print("Tuning:", opt.tuning)
    # print('MTL:', opt.mtl)
    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = U_PraNet().cuda()
    # model = PraNet().cuda()

    # model2 = vit_large(pretrained=True)

    # model2 = torch_model.vgg16(pretrained=True)
    # model2.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # model1 = TransFuse_L(pretrained=True)
    model1 = PolypPVT()
    # if opt.tuning:
    #     model1.load_state_dict(torch.load('./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth'))

    if opt.tuning and opt.mtl == 'stl':
        for param in model1.parameters():
            param.requires_grad = False

    model1.cuda()

    model2 = Discriminator()

    model2 = model2.cuda()

    models = {'Transfuse': model1,
              'Discriminator': model2}

    # models = {'Transfuse': TransFuse_L(pretrained=True).cuda(),
    #           'Discriminator': Discriminator().cuda()}

    # model_state = torch.load('PraNet-19.pth')
    # model.load_state_dict(model_state)############################################

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    if not opt.analytic:
        weight_methods_parameters = extract_weight_method_parameters_from_args()
        weight_method = WeightMethods(
            opt.mtl, n_tasks=2, device='cuda:0', **weight_methods_parameters[opt.mtl]
        )

    params = [p for v in models.values() for p in list(v.parameters())]
    # shared_parameters = [p for v in models.values() for p in list(v.parameters())]

    shared_parameters = [p for n, p in model1.named_parameters() if 'final_x' not in n and 'final_1' not in n and 'cls' not in n]
    task_specific_parameters = [p for n, p in model1.named_parameters() if "final_x" in n or 'final_1' in n or 'cls' in n]
    for n, p in model2.named_parameters():
        task_specific_parameters.append(p)

    # no=1
    # for n, p in model1.named_parameters():
    #     # if ("final_x" or 'final_1') in n:
    #     print(no)
    #     print(n)
    #     no+=1

    # shared_parameters = [print(n) for n, p in model1.named_parameters() if "final_x" in n or "final_1" in n]
    # print()
    # task_specific_parameters = [print(n) for n, p in model2.named_parameters()]

    # params = model.parameters()
    # params2 = model2.parameters()

    if not opt.analytic:
        # optimizer = torch.optim.Adam(
        optimizer = torch.optim.AdamW(
            [
                dict(params=params, lr=opt.lr, weight_decay=1e-4),
                dict(params=weight_method.parameters(), lr=0.025),
            ],
        )
    else:
        optimizer = torch.optim.Adam(
            [
                dict(params=params, lr=opt.lr, betas=(opt.beta1, opt.beta2)),
            ],
        )

    # optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
    # optimizer = PCGrad(optimizer)

    # d_optimizer = torch.optim.Adam(params2, opt.lr)

    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = SmoothCrossEntropy()

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    image_root_val = '{}/images/'.format(opt.val_path)
    gt_root_val = '{}/masks/'.format(opt.val_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    val_loader = get_loader(image_root_val, gt_root_val, batchsize=opt.batchsize, trainsize=opt.trainsize, phase='val')

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    train_d_loss_list = []
    val_d_loss_list = []
    best_loss = 100000
    best2_loss = 100000

    time_list = []

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)  ###################################
        # train(train_loader, model, optimizer, epoch)
        start_time = time.time()
        epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss = train(dataloaders_dict, models,
                                                                                             optimizer,
                                                                                             criterion,
                                                                                             epoch, best_loss,
                                                                                             best2_loss)
        print('-' * 20)
        print("学習時間: ", time.time() - start_time)
        time_list.append(time.time() - start_time)
        print('-' * 20)
        epoch_list.append(epoch)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        train_d_loss = train_d_loss.cpu().data.numpy()
        train_d_loss_list.append(train_d_loss)
        val_d_loss = val_d_loss.cpu().data.numpy()
        val_d_loss_list.append(val_d_loss)

        try:
            fig = plt.figure()
            plt.plot(epoch_list, train_loss_list, label='train_loss')
            plt.plot(epoch_list, val_loss_list, label='val_loss', linestyle="--")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.xlim(left=0)
            plt.legend(loc='upper right')
            fig.savefig(f"fig/{opt.train_save}_segmentation.png")
            plt.close(fig)

            fig2 = plt.figure()
            plt.plot(epoch_list, train_d_loss_list, label='train_d_loss', linestyle=":")
            plt.plot(epoch_list, val_d_loss_list, label='val_d_loss', linestyle="-.")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.xlim(left=0)
            plt.legend(loc='upper right')
            fig2.savefig(f"fig/{opt.train_save}_discriminator.png")
            plt.close(fig2)

        except:
            print('matplot processing failed')

    print("平均学習時間: ", sum(time_list) / len(time_list))
