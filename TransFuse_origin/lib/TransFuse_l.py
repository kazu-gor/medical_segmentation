import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.DeiT import deit_base_patch16_384 as deit_base
from lib.models_vit import vit_large_patch16 as vit_large
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import (Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear,
                      Softmax)
from torchvision.models import resnet50 as Resnet

# assert timm.__version__ == "0.3.2" # version check



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.0):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)  # ^bi

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in  # ^gi

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in  # ^ti
        fuse = self.residual(torch.cat([g, x, bp], 1))  # fi

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class AttnTransFuse_L(nn.Module):
    def __init__(
        self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False
    ):
        super(AttnTransFuse_L, self).__init__()

        # ============================================================
        self.Resnet_attn = Resnet()
        if pretrained:
            self.Resnet_attn.load_state_dict(
                torch.load("./weights/resnet50_a1_0-14fe96d1.pth")
            )
        self.Resnet_attn.fc = nn.Identity()
        self.Resnet_attn.layer4 = nn.Identity()

        self.transformer_attn = vit_large(pretrained=pretrained)

        self.up1_attn = Up(in_ch1=1024, out_ch=512)  # vit
        self.up2_attn = Up(512, 256)

        self.conv1x1_1_vit = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.conv1x1_1_cnn = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2_vit = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2_cnn = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3_vit = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3_cnn = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # ============================================================

        self.Resnet = Resnet()
        if pretrained:
            self.Resnet.load_state_dict(
                torch.load("./weights/resnet50_a1_0-14fe96d1.pth")
            )
        self.Resnet.fc = nn.Identity()
        self.Resnet.layer4 = nn.Identity()

        # self.transformer = deit_base(pretrained=pretrained)
        self.transformer = vit_large(pretrained=pretrained)
        # self.transformer = vit_large(pretrained=False)

        #############################################################################################
        self.up1 = Up(in_ch1=1024, out_ch=512)  # vit
        # self.up1 = Up(in_ch1=768, out_ch=512) # deit
        #############################################################################################
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # BiFusion

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # Transformer

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # Joint

        #############################################################################################
        self.up_c = BiFusion_block(
            ch_1=1024,
            ch_2=1024,
            r_2=4,
            ch_int=1024,
            ch_out=1024,
            drop_rate=drop_rate / 2,
        )  # top # vit
        # self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)  # top # deit
        #############################################################################################

        self.up_c_1_1 = BiFusion_block(
            ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2
        )  # mid
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(
            ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2
        )  # under
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, attn_map, labels=None):
        # ============================================================
        # bottom-up path for Attn
        x_b_attn = self.transformer_attn(
            attn_map
        )  # torch.Size([16, 3, 224, 224]) → torch.Size([16, 196, 1024])
        x_b_attn = torch.transpose(x_b_attn, 1, 2)  # torch.Size([16, 1024, 196])
        # x_b_attn = x_b_attn.view(x_b_attn.shape[0], -1, 14, 14)  # t0 # torch.Size([16, 1024, 14, 14]) # 224*224
        x_b_attn = x_b_attn.view(
            x_b_attn.shape[0], -1, 22, 22
        )  # t0 # torch.Size([16, 1024, 14, 14]) # 352*352

        x_b_attn = self.drop(x_b_attn)
        x_b_attn_1 = self.up1_attn(x_b_attn)  # t1 # torch.Size([16, 512, 28, 28])
        x_b_attn_1 = self.drop(x_b_attn_1)

        x_b_attn_2 = self.up2_attn(
            x_b_attn_1
        )  # transformer pred supervise here #t2 # torch.Size([16, 256, 56, 56])
        x_b_attn_2 = self.drop(x_b_attn_2)

        # top-down path for Attn

        x_u_attn = self.Resnet_attn.conv1(attn_map)  # torch.Size([16, 64, 112, 112])
        x_u_attn = self.Resnet_attn.bn1(x_u_attn)
        x_u_attn = self.Resnet_attn.relu(x_u_attn)
        x_u_attn = self.Resnet_attn.maxpool(x_u_attn)  # torch.Size([16, 64, 56, 56])

        x_u_attn_2 = self.Resnet.layer1(x_u_attn)  # g2 # torch.Size([16, 256, 56, 56])
        x_u_attn_2 = self.drop(x_u_attn_2)

        x_u_attn_1 = self.Resnet.layer2(
            x_u_attn_2
        )  # g1 # torch.Size([16, 512, 28, 28])
        x_u_attn_1 = self.drop(x_u_attn_1)

        x_u_attn = self.Resnet.layer3(x_u_attn_1)  # g0 # torch.Size([16, 1024, 14, 14])
        x_u_attn = self.drop(x_u_attn)

        # ============================================================

        # bottom-up path
        x_b = self.transformer(
            imgs
        )  # torch.Size([16, 3, 224, 224]) → torch.Size([16, 196, 1024])
        x_b = torch.transpose(x_b, 1, 2)  # torch.Size([16, 1024, 196])
        # x_b = x_b.view(x_b.shape[0], -1, 14, 14)  # t0 # torch.Size([16, 1024, 14, 14]) # 224*224
        x_b = x_b.view(
            x_b.shape[0], -1, 22, 22
        )  # t0 # torch.Size([16, 1024, 14, 14]) # 352*352

        x_b = self.drop(x_b)
        x_b_1 = self.up1(x_b)  # t1 # torch.Size([16, 512, 28, 28])
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(
            x_b_1
        )  # transformer pred supervise here #t2 # torch.Size([16, 256, 56, 56])
        x_b_2 = self.drop(x_b_2)

        # top-down path
        x_u = self.Resnet.conv1(imgs)  # torch.Size([16, 64, 112, 112])
        x_u = self.Resnet.bn1(x_u)
        x_u = self.Resnet.relu(x_u)
        x_u = self.Resnet.maxpool(x_u)  # torch.Size([16, 64, 56, 56])

        x_u_2 = self.Resnet.layer1(x_u)  # g2 # torch.Size([16, 256, 56, 56])
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.Resnet.layer2(x_u_2)  # g1 # torch.Size([16, 512, 28, 28])
        x_u_1 = self.drop(x_u_1)

        x_u = self.Resnet.layer3(x_u_1)  # g0 # torch.Size([16, 1024, 14, 14])
        x_u = self.drop(x_u)

        # ============================================================
        # Attention Map Integration

        # # adding * 0.5
        # x_u = x_u * 0.5 + x_u_attn * 0.5
        # x_u_1 = x_u_1 * 0.5 + x_u_attn_1 * 0.5
        # x_u_2 = x_u_2 * 0.5 + x_u_attn_2 * 0.5
        # x_b = x_b * 0.5 + x_b_attn * 0.5
        # x_b_1 = x_b_1 * 0.5 + x_b_attn_1 * 0.5
        # x_b_2 = x_b_2 * 0.5 + x_b_attn_2 * 0.5

        # concat channel dimension
        x_u_cat = torch.cat((x_u, x_u_attn), dim=1)
        x_u_1_cat = torch.cat((x_u_1, x_u_attn_1), dim=1)
        x_u_2_cat = torch.cat((x_u_2, x_u_attn_2), dim=1)
        x_b_cat = torch.cat((x_b, x_b_attn), dim=1)
        x_b_1_cat = torch.cat((x_b_1, x_b_attn_1), dim=1)
        x_b_2_cat = torch.cat((x_b_2, x_b_attn_2), dim=1)

        # 1x1 conv
        x_u = self.conv1x1_1_vit(x_u_cat)
        x_u_1 = self.conv1x1_2_vit(x_u_1_cat)
        x_u_2 = self.conv1x1_3_vit(x_u_2_cat)
        x_b = self.conv1x1_1_cnn(x_b_cat)
        x_b_1 = self.conv1x1_2_cnn(x_b_1_cat)
        x_b_2 = self.conv1x1_3_cnn(x_b_2_cat)

        # ============================================================

        # joint path
        x_c = self.up_c(
            x_u, x_b
        )  # biFusion pred here #f0 # torch.Size([16, 1024, 14, 14])
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)  # f1 # torch.Size([16, 512, 28, 28])
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)  # torch.Size([16, 512, 28, 28])
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)  # f2 # torch.Size([16, 256, 56, 56])
        x_c_2 = self.up_c_2_2(
            x_c_1, x_c_2_1
        )  # joint predict low supervise here # torch.Size([16, 256, 56, 56])

        # decoder part
        map_x = F.interpolate(
            self.final_x(x_c), scale_factor=16, mode="bilinear"
        )  # BiFusion pred map
        map_1 = F.interpolate(
            self.final_1(x_b_2), scale_factor=4, mode="bilinear"
        )  # Transformer pred map
        map_2 = F.interpolate(
            self.final_2(x_c_2), scale_factor=4, mode="bilinear"
        )  # Joint pred map
        return map_x, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


class TransFuse_L(nn.Module):
    def __init__(
        self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False
    ):
        super(TransFuse_L, self).__init__()

        self.Resnet = Resnet()
        if pretrained:
            self.Resnet.load_state_dict(
                torch.load("./weights/resnet50_a1_0-14fe96d1.pth")
            )
        self.Resnet.fc = nn.Identity()
        self.Resnet.layer4 = nn.Identity()

        # self.transformer = deit_base(pretrained=pretrained)
        self.transformer = vit_large(pretrained=pretrained)
        # self.transformer = vit_large(pretrained=False)

        #############################################################################################
        self.up1 = Up(in_ch1=1024, out_ch=512)  # vit
        # self.up1 = Up(in_ch1=768, out_ch=512) # deit
        #############################################################################################
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # BiFusion

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # Transformer

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False),
        )  # Joint

        #############################################################################################
        self.up_c = BiFusion_block(
            ch_1=1024,
            ch_2=1024,
            r_2=4,
            ch_int=1024,
            ch_out=1024,
            drop_rate=drop_rate / 2,
        )  # top # vit
        # self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)  # top # deit
        #############################################################################################

        self.up_c_1_1 = BiFusion_block(
            ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2
        )  # mid
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(
            ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2
        )  # under
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(
            imgs
        )  # torch.Size([16, 3, 224, 224]) → torch.Size([16, 196, 1024])
        x_b = torch.transpose(x_b, 1, 2)  # torch.Size([16, 1024, 196])
        # x_b = x_b.view(x_b.shape[0], -1, 14, 14)  # t0 # torch.Size([16, 1024, 14, 14]) # 224*224
        x_b = x_b.view(
            x_b.shape[0], -1, 22, 22
        )  # t0 # torch.Size([16, 1024, 14, 14]) # 352*352

        x_b = self.drop(x_b)
        x_b_1 = self.up1(x_b)  # t1 # torch.Size([16, 512, 28, 28])
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(
            x_b_1
        )  # transformer pred supervise here #t2 # torch.Size([16, 256, 56, 56])
        x_b_2 = self.drop(x_b_2)

        # top-down path
        x_u = self.Resnet.conv1(imgs)  # torch.Size([16, 64, 112, 112])
        x_u = self.Resnet.bn1(x_u)
        x_u = self.Resnet.relu(x_u)
        x_u = self.Resnet.maxpool(x_u)  # torch.Size([16, 64, 56, 56])

        x_u_2 = self.Resnet.layer1(x_u)  # g2 # torch.Size([16, 256, 56, 56])
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.Resnet.layer2(x_u_2)  # g1 # torch.Size([16, 512, 28, 28])
        x_u_1 = self.drop(x_u_1)

        x_u = self.Resnet.layer3(x_u_1)  # g0 # torch.Size([16, 1024, 14, 14])
        x_u = self.drop(x_u)

        # joint path
        x_c = self.up_c(
            x_u, x_b
        )  # biFusion pred here #f0 # torch.Size([16, 1024, 14, 14])
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)  # f1 # torch.Size([16, 512, 28, 28])
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)  # torch.Size([16, 512, 28, 28])
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)  # f2 # torch.Size([16, 256, 56, 56])
        x_c_2 = self.up_c_2_2(
            x_c_1, x_c_2_1
        )  # joint predict low supervise here # torch.Size([16, 256, 56, 56])

        # decoder part
        map_x = F.interpolate(
            self.final_x(x_c), scale_factor=16, mode="bilinear"
        )  # BiFusion pred map
        map_1 = F.interpolate(
            self.final_1(x_b_2), scale_factor=4, mode="bilinear"
        )  # Transformer pred map
        map_2 = F.interpolate(
            self.final_2(x_c_2), scale_factor=4, mode="bilinear"
        )  # Joint pred map
        return map_x, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        """
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        """
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(
        self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True
    ):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


if __name__ == "__main__":
    ras = TransFuse_S().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out = ras(input_tensor)
    print(out.shape())
