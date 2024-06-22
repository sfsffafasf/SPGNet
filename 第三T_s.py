import torch
import torch.nn as nn
# from mmcv.cnn import ConvAWS2d, constant_init
# from mmcv.ops.deform_conv import deform_conv2d
import torchvision.models as models
import scipy.stats as st
from torch.nn import functional as F
import numpy as np
import cv2
from torch.nn.parameter import Parameter
from backbone.mix_transformer import mit_b0, mit_b4
# from toolbox.models.DCTMO0.lv import GaborLayer
import time
# from toolbox.models.DCTMO0/bo.py
# from toolbox.model.cai.修layer import MultiSpectralAttentionLayer
import math
from toolbox.models.DCTMO0.bo import WaveAttention, SAN


# toolbox/models/DCTMO0/Lib0.py

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    gaussian_1D = np.linspace(-1, 1, k)
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


class CannyFilter1(nn.Module):
    def __init__(self, inc, k_gaussian=1, mu=0, sigma=3, use_cuda=True):
        super(CannyFilter1, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        gaussian_2D = torch.tensor(gaussian_2D, dtype=torch.float32).to(self.device)

        self.gaussian_filter = nn.Conv2d(inc, inc, kernel_size=k_gaussian, padding=k_gaussian // 2, bias=False)
        self.gaussian_filter.weight.data.copy_(gaussian_2D)

    def forward(self, img):
        # 2. 使用高斯滤波器平滑图像
        blurred_img = self.gaussian_filter(img)
        return blurred_img


# 卷积
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        # self.conv = Dynamic_conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=dilation, bias=False)  ##改了动态卷积
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=(3 - 1) // 2)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = self.sigmoid(self.conv(avg_out+max_out))

        return out


class Ca_C(nn.Module):
    def __init__(self, channel, reduction=8):
        super(Ca_C, self).__init__()
        # c2wh = dict([(64, 104), (128, 52), (320, 26), (512, 13)])
        self.con1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=3, padding=1,
                              dilation=1, bias=False)
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        # self.Channel_Att = Channel_Att(channel)
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 1, 2, 3)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1)

        m = self.reconstruct(x_out11, x_out2)*x + x

        return m


class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()
        self.conv1_1 = BasicConv2d(in_channel, in_channel, 3, 1, 1, 1)
        self.conv2_1 = BasicConv2d(in_channel, in_channel, 3, 1, 4, 4)
        self.conv3_1 = BasicConv2d(in_channel, in_channel, 3, 1, 8, 8)
        self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        # buffer_1.append(self.CannyFilter1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        out = self.relu(buffer_1 + x)
        return out


# BatchNorm2d = nn.BatchNorm2d
# BatchNorm1d = nn.BatchNorm1d
class MFA1(nn.Module):
    def __init__(self, img1channel, img2channel):
        super(MFA1, self).__init__()
        # b0 32, 64, 160, 256    64, 128, 320, 512
        self.Ca_C = Ca_C(channel=img1channel)
        # self.sof = nn.Softmax(dim=1)
        self.CannyFilter1 = CannyFilter1(img1channel)
        # self.SAN = SAN(img1channel)
        # self.WaveAttention = WaveAttention(img1channel)
        self.layer_img = nn.Sequential(nn.Conv2d(img1channel, img2channel, kernel_size=3, stride=1, padding=1),
                                       nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.BatchNorm2d(img2channel), nn.LeakyReLU(inplace=True))

    def forward(self, img1, dep1):
        ################################[2, 32, 28, 28]
        """
        :param ful: 2, 64, 52
        :param img1: 2, 32, 104
        :param dep1:
        :param img: 2,64,52
        :param dep:
        :return:
        """

        rd = img1 + dep1
        out = rd + self.CannyFilter1(rd)
        out = self.Ca_C(out)
        out = self.layer_img(out)
        return out


class MFA(nn.Module):
    def __init__(self, img1channel, img2channel):
        super(MFA, self).__init__()
        # b0 32, 64, 160, 256    64, 128, 320, 512
        # c2wh = dict([(32, 104), (64, 52), (160, 26), (256, 13)])
        c2wh = dict([(64, 104), (128, 52), (320, 26), (512, 13)])
        # self.r_ca = Ca_block(channel=img1channel)
        # self.sof = nn.Softmax(dim=1)
        self.er = ER(in_channel=img1channel)
        # self.SAN = SAN(img1channel)
        self.WaveAttention = WaveAttention(img1channel)
        self.layer_img = nn.Sequential(nn.Conv2d(img1channel, img2channel, kernel_size=3, stride=1, padding=1),
                                       nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.BatchNorm2d(img2channel), nn.LeakyReLU(inplace=True))

    def forward(self, img1, dep1):
        ################################[2, 32, 28, 28]
        """
        :param ful: 2, 64, 52
        :param img1: 2, 32, 104
        :param dep1:
        :param img: 2,64,52
        :param dep:
        :return:
        """
        # img2 = img1+img1*dep1
        # dep2 = dep1+dep1*dep1
        # weighting = self.layer_img(torch.cat([img2, dep2], dim=1))
        rd = img1 + dep1
        out = rd + self.WaveAttention(rd)
        out = self.er(out)
        out = self.layer_img(out)
        return out


class Decode(nn.Module):
    def __init__(self, in_dim):
        super(Decode, self).__init__()
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_dim[3], in_dim[2], kernel_size=3, stride=1, padding=1),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.BatchNorm2d(in_dim[2]), nn.LeakyReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_dim[3], in_dim[1], kernel_size=3, stride=1, padding=1),
                                     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                     nn.BatchNorm2d(in_dim[1]), nn.LeakyReLU(inplace=True))
        self.conv3_0 = nn.Sequential(nn.Conv2d(in_dim[3], in_dim[0], kernel_size=3, stride=1, padding=1),
                                     nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                                     nn.BatchNorm2d(in_dim[0]), nn.LeakyReLU(inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_dim[2], in_dim[1], kernel_size=3, stride=1, padding=1),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.BatchNorm2d(in_dim[1]), nn.LeakyReLU(inplace=True))
        self.conv1_0 = nn.Sequential(nn.Conv2d(in_dim[1], in_dim[0], kernel_size=3, stride=1, padding=1),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.BatchNorm2d(in_dim[0]), nn.LeakyReLU(inplace=True))

    def forward(self, ful_3, ful_2, ful_1, ful_0):
        m2 = self.conv3_2(ful_3) + ful_2
        m1 = self.conv2_1(m2) + self.conv3_1(ful_3) + ful_1
        m0 = self.conv1_0(m1) + self.conv3_0(ful_3) + ful_0
        return m0

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class GNN(nn.Module):
    def __init__(self, n_F, n_gnn=1):
        '''
            Time complexity: O(NNF+NFF)
        '''
        super().__init__()
        self.relu = nn.ReLU()
        # self.n_gnn = n_gnn
        self.W = nn.Parameter(torch.empty((n_F, n_F)))
        self.bn = LayerNorm(n_F)
        # self.so = nn.Softmax()
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X, reverse=False):

        n, c, h, w = X.size()
        X = X.view(n * c, h, w)
        A = torch.transpose(X, 2, 1)
        A = F.softmax(A, 2, torch.float32)
        if reverse: A = 1.0 - A
        node = torch.matmul(A, self.bn(X))
        X = self.relu(torch.matmul(node, self.W))
        return X.view(n, c, h, w)


class similarity2(nn.Module):
    def __init__(self, in_channels):
        super(similarity2, self).__init__()
        self.C = in_channels // 4
        self.S = in_channels
        self.m1 = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)
        self.m2 = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)
        self.conv_2 = BasicConv2d(self.S, in_channels, 1, 1)

    def forward(self, x):
        batch, c, h, w = x.size()
        L = h * w
        B = self.m1(x).view(-1, self.C, L) #N*L
        B2 = self.m2(x).view(-1, self.S, L)  #S*L
        B2 = torch.transpose(B2, 1, 2)       #L*S
        V = torch.bmm(B, B2) / L  #  #N*S
        y = torch.bmm(torch.transpose(B, 1, 2), V)
        y = y.view(-1, self.S, h, w)
        y = self.conv_2(y)

        return y


class FoldUnfoldModule(nn.Module):
    def __init__(self,channels,context_size=3):
        super(FoldUnfoldModule, self).__init__()

        self.context_size = context_size
        self.pad = context_size // 2

        self.GNN1 = GNN(26)
        in_channels = context_size * context_size
        self.conv = BasicConv2d(in_channels, channels, 3, padding=1, dilation=1)
        self.Interv2 = similarity2(channels)

    def self_similarity(self, feature_normalized):
        """
        计算自相似度特征。
        """
        b, c, h, w = feature_normalized.size()
        feature_pad = F.pad(feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        output = torch.zeros(
            [self.context_size * self.context_size, b, h, w],
            dtype=feature_normalized.dtype,
            requires_grad=feature_normalized.requires_grad,
        )
        if feature_normalized.is_cuda:
            output = output.cuda(feature_normalized.get_device())
        # print(output.shape,feature_normalized.shape)
        for i in range(self.context_size):
            for j in range(self.context_size):
                if i == 0 and j == 0:
                    continue  # 忽略中心点自身
                roi = feature_pad[:, :, i:(h + i), j:(w + j)]
                # print(roi.shape,output.shape)
                output[i * self.context_size + j] = (roi * feature_normalized).sum(dim=1)
        output = output.transpose(0, 1).contiguous()
        return output
    def forward(self, x):
        # print(self.self_similarity(x).shape)
        m = self.conv(self.self_similarity(x))
        xm = m*x
        # print(xm.shape)
        return self.GNN1(self.Interv2(x)+xm)+x




"""
rgb和d分别与融合的做乘法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
rgb和d分别与融合的做加法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
输出就是融合
"""


####################################################自适应1,2,3,6###########################

class LiSPNetx22(nn.Module):
    def __init__(self, channel=32):
        super(LiSPNetx22, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # Backbone model   In
        # Backbone model32, 64, 160, 256
        # self.layer_dep0 = nn.Conv2d(3, 3, kernel_size=1)
        # res_channels = [32, 64, 160, 256, 256]
        # channels = [64, 128, 256, 512, 512]
        self.resnet = mit_b4()
        self.resnet_depth = mit_b4()
        self.resnet.init_weights("/home/noone/桌面/李超/backbone/mit_b4.pth")
        self.resnet_depth.init_weights("/home/noone/桌面/李超/backbone/mit_b4.pth")
        ###############################################
        # funsion encoders #
        ## rgb64, 128, 320, 512
        # channels = [32, 64, 160, 256]
        channels = [64, 128, 320, 512]
        self.ful_3 = MFA(channels[3], channels[2])
        self.ful_2 = MFA(channels[2], channels[1])
        self.ful_1 = MFA1(channels[1], channels[0])
        self.ful_0 = MFA1(channels[0], 16)
        # self.GCM_3 = GCM(channels[3], channels[2])
        # self.GCM_2 = GCM(channels[2], channels[1])
        # self.GCM_1 = GCM(channels[1], channels[0])
        # self.GCM_0 = GCM(channels[0],          16)
        self.conv_out1 = nn.Conv2d(16, 1, 1)
        self.conv_out2 = nn.Conv2d(channels[0], 1, 1)
        self.conv_out3 = nn.Conv2d(channels[1], 1, 1)
        self.conv_out4 = nn.Conv2d(channels[2], 1, 1)
        # chs = [16, 32, 64, 160]
        chs = [16, 64, 128, 320]
        self.Decode = Decode(chs)
        self.FoldUnfoldModule = FoldUnfoldModule(channels[2])
        # self.BidirectionalAttention = BidirectionalAttention(channels[3])
        # self.gamma1 = nn.Parameter(torch.zeros(1))

        # self.UnetD = UnetD(channelout)

    def forward(self, imgs, depths):
        # depths = imgs
        img_0, img_1, img_2, img_3 = self.resnet.forward_features(imgs)
        # print(img_0.shape, img_1.shape, img_2.shape, img_3.shape)
        ####################################################
        ## decoder rgb     ful_2.shape[2, 256, 14, 14]   img_3.shape [2, 256, 7, 7]
        ####################################################
        dep_0, dep_1, dep_2, dep_3 = self.resnet_depth.forward_features(depths)
        OUT_3 = self.ful_3(img_3, dep_3)
        OUT_2 = self.ful_2(img_2, dep_2)
        OUT_1 = self.ful_1(img_1, dep_1)
        OUT_0 = self.ful_0(img_0, dep_0)

        OUT_3 = self.FoldUnfoldModule(OUT_3)
        # print(OUT_3.shape,OUT_2.shape,OUT_1.shape,OUT_0.shape,)
        out = self.Decode(OUT_3, OUT_2, OUT_1, OUT_0)
        ########2，256，13      32,208     64,104
        ful1 = self.conv_out1(self.upsample_2(out))
        ful2 = self.conv_out2(self.upsample_4(OUT_1))
        ful3 = self.conv_out3(self.upsample_8(OUT_2))
        # ful4 = self.conv_out4(self.upsample_16(ful_3))
        return ful1, ful2, ful3  # ,ful4,OUT_3,OUT_2,OUT_1,OUT_0,img_3,dep_3,img_2, dep_2, img_1, dep_1, img_0, dep_0,ful_3,ful_2,ful_1,ful_0


if __name__ == "__main__":
    rgb = torch.randn(2, 3, 416, 416).cuda()
    t = torch.randn(2, 3, 416, 416).cuda()
    model = LiSPNetx22().cuda()
    out = model(rgb, t)
    for i in range(len(out)):
        print(out[i].shape)  # Flops:6.48 GMac  Params:13.88 M

    # from toolbox import compute_speed/
# Speed Time: 28.42 ms / iter   FPS: 35.18Flops:  4.57 GMac
# # Params: 10.03 M
# def contrastive_loss(out, out_aug, batch_size=2, hidden_norm=False, temperature=1.0):
#     if hidden_norm:
#         out = F.normalize(out, dim=-1)
#         out_aug = F.normalize(out_aug, dim=-1)
#     INF = float('inf')
#     labels = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size * 2)  # [batch_size,2*batch_size]
#     masks = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size)  # [batch_size,batch_size]
#     logits_aa = torch.matmul(out, out.transpose(0, 1)) / temperature  # [batch_size,batch_size]
#     logits_bb = torch.matmul(out_aug, out_aug.transpose(0, 1)) / temperature  # [batch_size,batch_size]
#     logits_aa = logits_aa - masks * INF  # remove the same samples in out
#     logits_bb = logits_bb - masks * INF  # remove the same samples in out_aug
#     logits_ab = torch.matmul(out, out_aug.transpose(0, 1)) / temperature
#     logits_ba = torch.matmul(out_aug, out.transpose(0, 1)) / temperature
#     loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
#     loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
#     loss = loss_a + loss_b
#     return loss
# class OFD(nn.Module):
#     '''
#     A Comprehensive Overhaul of Feature Distillation
#     http://openaccess.thecvf.com/content_ICCV_2019/papers/
#     Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
#     '''
#
#     def __init__(self, in_channels, out_channels):
#         super(OFD, self).__init__()
#         self.connector = nn.Sequential(*[
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ])
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, fm_s, fm_t):
#         margin = self.get_margin(fm_t)
#         fm_t = torch.max(fm_t, margin)
#         fm_s = self.connector(fm_s)
#
#         mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
#         loss = torch.mean((fm_s - fm_t) ** 2 * mask)
#
#         return loss
#
#     def get_margin(self, fm, eps=1e-6):
#         mask = (fm < 0.0).float()
#         masked_fm = fm * mask
#
#         margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)
#
#         return margin


# from toolbox import compute_speed
# from ptflops import get_model_complexity_info
# with torch.cuda.device(0):
#     net = LiSPNetx22().cuda()
#     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
#     compute_speed(net, input_size=(1, 3, 416, 416), iteration=500)
#     print('Flops:'+flops)
#     print('Params:'+params)
# print(a.shape)
# Flops:33.52 GMac
# Params:190.26 M
# print(a[1].shape)
# print(a[2].shape)
# print(a[3].shape)Elapsed Time: [17.54 s / 500 iter]
# Speed Time: 35.08 ms / iter   FPS: 28.51
# compute_speed(net,input_size=(1, 3, 416, 416), iteration=500)    此处加了se注意力