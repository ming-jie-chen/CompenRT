import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_tps
from torch.nn import init
from dysample import DySample
#
from carafe import CARAFE
# from PSA import BasicRFB
from torch.nn.parameter import Parameter
import common
from torchsummary import summary
import math
from einops import rearrange
class PA(nn.Module):  # Pixel Attention
    '''
    '''

    def __init__(self, channel):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.sig(c1_)

        return x * c1


class LANet(nn.Module):
    def __init__(self):
        super(LANet, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.scale = 4

        self.simplified = False
        self.unshuffle = nn.PixelUnshuffle(self.scale)
        self.pa1 = PA(48)
        self.pa2 = PA(48)

        # siamese encoder
        self.conv1 = nn.Conv2d(3 * self.scale * self.scale, 48, 3, 2, 1)
        self.conv2 = nn.Conv2d(48, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.transConv2 = nn.ConvTranspose2d(64, 48, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(48, 3 * self.scale * self.scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(self.scale)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(48, 48, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(48, 48, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(48, 48, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(48, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases

    def simplify(self, s):
        s = (self.unshuffle(s))
        s = self.pa1(s)

        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = (self.conv1(s))
        s = self.pa2(s)

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a surface image
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            s = self.unshuffle(s)
            s = self.pa1(s)

            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = (self.conv1(s))
            s = self.pa2(s)

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        x = self.unshuffle(x)
        x = self.pa1(x)

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 - res1_s

        x = (self.conv1(x))
        x = self.pa2(x)

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 = res2 - res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2)

        x = self.relu(self.transConv2(x) + res1)
        x = (self.conv6(x))
        x = self.shuffle(x)
        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)

        return x
#
class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=13, padding=6),
            nn.Sigmoid())
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        self.apply(_initialize_weights)

    def forward(self, x):
        x = self.conv_du(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid())
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale a undpscale --> channel weight
        self.conv_du = nn.Sequential(
            # channel // reduction，输出降维，即论文中的1x1xC/r
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # channel，输出升维，即论文中的1x1xC
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        self.apply(_initialize_weights)

    def forward(self, x):
        # b, c, _, _ = x.size()
        # x = self.avg_pool(x).view(b, c)
        # x = self.fc(x).view(b, c, 1, 1)

        x = self.avg_pool(x)
        x = self.conv_du(x)
        return x

class improvePA(nn.Module):#Pixel Attention
    '''
    '''
    def __init__(self,channel,reduction=16):
        super(improvePA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channel,1,kernel_size=3,padding=1),
            # nn.Conv2d(channel, 1, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(1, 1, kernel_size=13, padding=6),
            # nn.ReLU(),
            # nn.Conv2d(1, 1, kernel_size=13, padding=6),
            nn.Sigmoid())
        self.ca= nn.Sequential(
            nn.AdaptiveAvgPool2d(1),# global average pooling: feature --> point
            # feature channel downscale a undpscale --> channel weight
            # channel // reduction，输出降维，即论文中的1x1xC/r
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            # channel，输出升维，即论文中的1x1xC
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )
        # self.Spatialattention=SpatialAttention(channel)
        # self.Channelattention=ChannelAttention(channel)
    def forward(self, x):
        Ms=self.sa(x)
        Mc=self.ca(x)
        Mp=Mc*Ms
        return Mp

#
# class improvePA(nn.Module):  # Pixel Attention
#     '''
#     '''
#
#     def __init__(self, channel, reduction=16):
#         super(improvePA, self).__init__()
#         self.sa = nn.Sequential(
#             # nn.Conv2d(channel,1,kernel_size=3,padding=1),
#             nn.Conv2d(channel, 1, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=13, padding=6),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=13, padding=6),
#             nn.Sigmoid())
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # global average pooling: feature --> point
#             # feature channel downscale a undpscale --> channel weight
#             # channel // reduction，输出降维，即论文中的1x1xC/r
#             nn.Conv2d(channel, channel // reduction, 1, padding=0),
#             nn.ReLU(inplace=True),
#             # channel，输出升维，即论文中的1x1xC
#             nn.Conv2d(channel // reduction, channel, 1, padding=0),
#             nn.Sigmoid()
#         )
#         # self.Spatialattention=SpatialAttention(channel)
#         # self.Channelattention=ChannelAttention(channel)
#
#     def forward(self, x):
#         out = x * self.ca(x)
#         y = out * self.sa(out)
#         return y
#
# class PANet(nn.Module):
#     def __init__(self):
#         super(PANet, self).__init__()
#         self.name = self.__class__.__name__
#         self.relu = nn.ReLU()
#         self.scale = 4
#
#         self.simplified = False
#         self.unshuffle = nn.PixelUnshuffle(self.scale)
#         self.conv1=nn.Conv2d(3,32,3,2,1)
#         self.norm1=nn.GroupNorm(num_groups=16, num_channels=32)
#         self.pa1=PA(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
#         self.conv3=nn.Conv2d(64,128,3,1,1)
#         self.conv4=nn.Conv2d(128,256,3,1,1)
#
#         self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
#         self.conv6 = nn.Conv2d(128, 64, 3, 1, 1)
#
#         # self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
#
#         # surface image feature extraction branch
#         self.conv1_s = nn.Conv2d(3, 32, 3, 2, 1)
#         self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
#         self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)
#         self.transConv1 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
#         # skip layers (see s3 in forward)
#         self.skipConv1 = nn.Sequential(
#             nn.Conv2d(3, 3, 3, 1, 1),
#             self.relu,
#             nn.Conv2d(3, 3, 3, 1, 1),
#             self.relu,
#             nn.Conv2d(3, 3, 3, 1, 1),
#             self.relu
#         )
#         self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
#         self.skipConv3 = nn.Conv2d(64, 128, 1, 1, 0)
#         # # s1
#         self.skipConv11 = nn.Conv2d(64, 64, 1, 1, 0)
#         self.skipConv12 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.skipConv13 = nn.Conv2d(64, 64, 3, 1, 1)
#         #
#         # # s2
#         # self.skipConv21 = nn.Conv2d(128, 128, 1, 1, 0)
#         # self.skipConv22 = nn.Conv2d(128, 128, 3, 1, 1)
#         # self.carafe=CARAFE(48,3)
#         # stores biases of surface feature branch (net simplification)
#         self.register_buffer('res1_s_pre', None)
#         self.register_buffer('res2_s_pre', None)
#         self.register_buffer('res3_s_pre', None)
#
#         # initialization function, first checks the module type,
#         def _initialize_weights(m):
#             if type(m) == nn.Conv2d:
#                 nn.init.kaiming_normal_(m.weight)
#
#         self.apply(_initialize_weights)
#
#     # simplify trained model by trimming surface branch to biases
#
#     def simplify(self, s):
#         s = (self.unshuffle(s))  # Downsampler
#         s = self.pa1(s)
#
#         res1_s = self.relu(self.skipConv11(s))
#         res1_s = self.relu(self.skipConv12(res1_s))
#         res1_s = self.skipConv13(res1_s)
#         self.res1_s_pre = res1_s  # 橙色的skip线
#
#         s = (self.conv1(s))
#         s = self.pa2(s)
#
#         res2_s = self.skipConv21(s)
#         res2_s = self.relu(res2_s)
#         res2_s = self.skipConv22(res2_s)
#         self.res2_s_pre = res2_s  # 绿色的skip线
#
#         s = self.relu(self.conv2(s))
#         s = self.relu(self.conv3(s))
#         self.res3_s_pre = s  # 紫色的skip线
#
#         self.res1_s_pre = self.res1_s_pre.squeeze()
#         self.res2_s_pre = self.res2_s_pre.squeeze()
#         self.res3_s_pre = self.res3_s_pre.squeeze()
#
#         self.simplified = True
#
#     # x is the input uncompensated image, s is a surface image
#     def forward(self, x, s):
#         # surface feature extraction
#
#         # alternate between surface and image branch
#         if self.simplified:
#             res1_s = self.res1_s_pre
#             res2_s = self.res2_s_pre
#             res3_s = self.res3_s_pre
#         else:
#             s=self.conv1(s)
#             s=self.norm1(s)
#             s = self.pa1(s)
#             s=self.conv2(s)
#             res1_s = self.relu(self.skipConv11(s))
#             res1_s = self.relu(self.skipConv12(res1_s))
#             res1_s = self.skipConv13(res1_s)
#             s=self.conv3(s)
#             res2_s = self.skipConv21(s)
#             res2_s = self.relu(res2_s)
#             res2_s = self.skipConv22(res2_s)
#             s=self.conv4(s)
#             res3_s=s
#
#
#
#
#             s = self.relu(self.conv3(s))
#             res3_s = self.relu(self.conv4(s))
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.pa1(x)
#         res1 = self.skipConv1(x)
#         # x = self.unshuffle(x)
#         x=self.conv1(x)
#         x=self.norm1(x)
#         x = self.pa1(x)
#
#         res1 = self.relu(self.skipConv11(x))
#         res1 = self.relu(self.skipConv12(res1))
#         res1 = self.skipConv13(res1)
#         res1 = res1 - res1_s
#
#         x = (self.conv2(x))
#         x = self.norm2(x)
#         x = self.pa2(x)
#
#         res2 = self.skipConv21(x)
#         res2 = self.relu(res2)
#         res2 = self.skipConv22(res2)
#         res2 = res2 - res2_s
#
#         x = self.relu(self.conv3(x))
#
#         x = self.relu(self.conv4(x))
#         x = x - res3_s  # s3
#
#         x = self.relu(self.conv5(x))
#         x = self.relu(self.conv6(x))
#
#         x = self.relu(self.transConv1(x) + res2)
#
#         x = self.relu(self.transConv2(x) + res1)
#         x = (self.conv7(x))
#         x = self.shuffle(x)
#         # x=self.dysample(x)
#         # x=self.conv7(x)
#         x = torch.clamp(x, max=1)
#         x = torch.clamp(x, min=0)
#
#         return x

#
# class PANet(nn.Module):
#     def __init__(self):
#         super(PANet, self).__init__()
#         self.name = self.__class__.__name__
#         self.relu = nn.ReLU()
#         self.scale = 4
#
#         self.simplified = False
#         self.unshuffle = nn.PixelUnshuffle(self.scale)
#         self.pa1 = PA(48)
#         self.pa2 = PA(48)
#         self.pa3 = improvePA(32)
#         # siamese encoder
#         self.conv1 = nn.Conv2d(3 * self.scale * self.scale, 48, 3, 2, 1)
#         self.conv2 = nn.Conv2d(48, 64, 3, 2, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
#         self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
#         self.conv = nn.Conv2d(3, 48, 3, 4, 1)
#         self.norm = nn.GroupNorm(num_groups=24, num_channels=48)
#         self.norm1 = nn.GroupNorm(num_groups=16, num_channels=32)
#         # transposed conv
#         self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
#         self.transConv2 = nn.ConvTranspose2d(64, 48, 2, 2, 0)
#
#         # output layer
#         self.conv6 = nn.Conv2d(48, 3 * self.scale * self.scale, 3, 1, 1)
#         self.shuffle = nn.PixelShuffle(self.scale)
#         # self.conv=BasicRFB(48,48)
#         self.carafe = CARAFE(32, 3)
#         self.dysample = DySample(48)
#         self.conv7 = nn.Conv2d(48, 3, 1)
#         self.conv8 = nn.Conv2d(48, 32, 3, 1, 1)
#         # skip layers (see s3 in forward)
#         # s1
#         self.skipConv11 = nn.Conv2d(48, 48, 1, 1, 0)
#         self.skipConv12 = nn.Conv2d(48, 48, 3, 1, 1)
#         self.skipConv13 = nn.Conv2d(48, 48, 3, 1, 1)
#
#         # s2
#         self.skipConv21 = nn.Conv2d(48, 64, 1, 1, 0)
#         self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)
#
#         # stores biases of surface feature branch (net simplification)
#         self.register_buffer('res1_s_pre', None)
#         self.register_buffer('res2_s_pre', None)
#         self.register_buffer('res3_s_pre', None)
#
#         # initialization function, first checks the module type,
#         def _initialize_weights(m):
#             if type(m) == nn.Conv2d:
#                 nn.init.kaiming_normal_(m.weight)
#
#         self.apply(_initialize_weights)
#
#     # simplify trained model by trimming surface branch to biases
#
#     def simplify(self, s):
#         s = (self.unshuffle(s))  # Downsampler
#         s = self.pa1(s)
#
#         res1_s = self.relu(self.skipConv11(s))
#         res1_s = self.relu(self.skipConv12(res1_s))
#         res1_s = self.skipConv13(res1_s)
#         self.res1_s_pre = res1_s  # 橙色的skip线
#
#         s = (self.conv1(s))
#         s = self.pa2(s)
#
#         res2_s = self.skipConv21(s)
#         res2_s = self.relu(res2_s)
#         res2_s = self.skipConv22(res2_s)
#         self.res2_s_pre = res2_s  # 绿色的skip线
#
#         s = self.relu(self.conv2(s))
#         s = self.relu(self.conv3(s))
#         self.res3_s_pre = s  # 紫色的skip线
#
#         self.res1_s_pre = self.res1_s_pre.squeeze()
#         self.res2_s_pre = self.res2_s_pre.squeeze()
#         self.res3_s_pre = self.res3_s_pre.squeeze()
#
#         self.simplified = True
#
#     # x is the input uncompensated image, s is a surface image
#     def forward(self, x, s):
#         # surface feature extraction
#
#         # alternate between surface and image branch
#         if self.simplified:
#             res1_s = self.res1_s_pre
#             res2_s = self.res2_s_pre
#             res3_s = self.res3_s_pre
#         else:
#             s = self.unshuffle(s)
#
#             s = self.pa1(s)
#
#             res1_s = self.relu(self.skipConv11(s))
#             res1_s = self.relu(self.skipConv12(res1_s))
#             res1_s = self.skipConv13(res1_s)
#
#             s = (self.conv1(s))
#
#             s = self.pa2(s)
#
#             res2_s = self.skipConv21(s)
#             res2_s = self.relu(res2_s)
#             res2_s = self.skipConv22(res2_s)
#
#             s = self.relu(self.conv2(s))
#             res3_s = self.relu(self.conv3(s))
#
#         x = self.unshuffle(x)
#
#         x = self.pa1(x)
#
#         res1 = self.relu(self.skipConv11(x))
#         res1 = self.relu(self.skipConv12(res1))
#         res1 = self.skipConv13(res1)
#         res1 = res1 - res1_s
#
#         x = (self.conv1(x))
#
#         x = self.pa2(x)
#
#         res2 = self.skipConv21(x)
#         res2 = self.relu(res2)
#         res2 = self.skipConv22(res2)
#         res2 = res2 - res2_s
#
#         x = self.relu(self.conv2(x))
#
#         x = self.relu(self.conv3(x))
#         x = x - res3_s  # s3
#
#         x = self.relu(self.conv4(x))
#         x = self.relu(self.conv5(x))
#
#         x = self.relu(self.transConv1(x) + res2)
#
#         x = self.relu(self.transConv2(x) + res1)
#         x = (self.conv6(x))
#         x = self.shuffle(x)
#         # x = self.carafe(x)
#         # x=self.dysample(x)
#         # x=self.conv7(x)
#         x = torch.clamp(x, max=1)
#         x = torch.clamp(x, min=0)
#
#         return x
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self,num_high=1):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2),mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image
# CompenNeSt (journal version)
class CompenNeSt(nn.Module):
    def __init__(self):
        super(CompenNeSt, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()

        self.simplified = False

        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv11=nn.Conv2d(48,64,1,1,0)
        self.conv12=nn.Conv2d(48,3,1,1,0)
        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s,high1,high2):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))
        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 =res1- res1_s

        x = self.relu(self.conv1(x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 =res2- res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x =x- res3_s # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2+self.conv11(high2))
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x)+res1+self.conv12(high1))

        x = torch.clamp(x, max=1)

        return x





class Trans_high(nn.Module):
    def __init__(self):
        super(Trans_high, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(),
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(),
            nn.Conv2d(48,48,3,1,1),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(48,48,3,1,1),
            nn.ReLU(),
            nn.Conv2d(48,48,3,1,1),
        )
        # self.conv1=nn.Conv2d(48,48,3,1,1)
        # self.conv2=nn.Conv2d(48,48,3,1,1)
        # self.conv3=nn.Conv2d(48,48,3,1,1)
        # self.conv4=nn.Conv2d(48,48,3,1,1)
        # self.conv5=nn.Conv2d(48,48,3,1,1)
        # self.conv1_s = nn.Conv2d(48, 48, 3, 1, 1)
        # self.conv2_s = nn.Conv2d(48, 48, 3, 1, 1)
        # self.conv3_s = nn.Conv2d(48, 48, 3, 1, 1)
        # self.conv4_s = nn.Conv2d(48, 48, 3, 1, 1)
        # self.conv5_s = nn.Conv2d(48, 48, 3, 1, 1)
        self.relu=nn.ReLU()
        self.unshuffle= nn.PixelUnshuffle(4)
        self.shuffle=nn.PixelShuffle(4)
    def forward(self, high_x, high_s):
        pyr_result = []
        high_x1=self.unshuffle(high_x[-2])
        high_s1=self.unshuffle(high_s[-2])
        high_s1=self.conv1(high_s1)


        high_x1 =self.conv1(high_x1)
        res1=high_x1-high_s1
        high_x1=self.shuffle(self.relu(res1))


        high_s2 = self.unshuffle(high_s[-3])
        high_s2=self.conv(high_s2)

        high_x2 = self.unshuffle(high_x[-3])
        high_x2=self.conv(high_x2)
        res2=high_x2-high_s2
        high_x2 = self.shuffle(self.relu(res2))

        pyr_result.append(high_x2)
        pyr_result.append(high_x1)

        return pyr_result,res2,res1
class PANet(nn.Module):
    def __init__(self,conv=common.default_conv):
        super(PANet, self).__init__()
        self.name = self.__class__.__name__

        self.relu=nn.ReLU()
        act = nn.ReLU(True)  # 激活函数
        self.simplified = False
        self.lap_pyramid = Lap_Pyramid_Conv(2)
        self.CompenNest=CompenNeSt()
        self.trans_high=Trans_high()
        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    #
    # def simplify(self, s):
    #     s = (self.unshuffle(s))  # Downsampler
    #     s = self.pa1(s)
    #
    #     res1_s = self.relu(self.skipConv11(s))
    #     res1_s = self.relu(self.skipConv12(res1_s))
    #     res1_s = self.skipConv13(res1_s)
    #     self.res1_s_pre = res1_s  # 橙色的skip线
    #
    #     s = (self.conv1(s))
    #     s = self.pa2(s)
    #
    #     res2_s = self.skipConv21(s)
    #     res2_s = self.relu(res2_s)
    #     res2_s = self.skipConv22(res2_s)
    #     self.res2_s_pre = res2_s  # 绿色的skip线
    #
    #     s = self.relu(self.conv2(s))
    #     s = self.relu(self.conv3(s))
    #     self.res3_s_pre = s  # 紫色的skip线
    #
    #     self.res1_s_pre = self.res1_s_pre.squeeze()
    #     self.res2_s_pre = self.res2_s_pre.squeeze()
    #     self.res3_s_pre = self.res3_s_pre.squeeze()
    #
    #     self.simplified = True

    # x is the input uncompensated image, s is a surface image
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            pyr_s=self.lap_pyramid.pyramid_decom(img=s)
        pyr_A=self.lap_pyramid.pyramid_decom(img=x)

        pyr_A_trans,res2,res1=self.trans_high(pyr_A,pyr_s)
        compen = self.CompenNest(pyr_A[-1], pyr_s[-1],res2,res1)
        # compen_up1=nn.functional.interpolate(compen, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        pyr_A_trans.append(compen)
        x= self.lap_pyramid.pyramid_recons(pyr_A_trans)
        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)

        return x





class GridRefine(nn.Module):
    def __init__(self):
        super(GridRefine, self).__init__()
        self.name = self.__class__.__name__
        # grid refinement net
        self.conv1 = nn.Conv2d(2, 32, 7, 4, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv2f = nn.Conv2d(32, 32, 1, 1, 0)
        self.conv31 = nn.Conv2d(64, 64, 7, 4, 3)
        self.conv32 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv33 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3f = nn.Conv2d(64, 64, 1, 1, 0)
        self.trans1 = nn.ConvTranspose2d(32, 2, 4, 4, 0)
        self.trans2 = nn.ConvTranspose2d(64, 64, 4, 4, 0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()


        # self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64)
        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        self.apply(_initialize_weights)

    def forward(self, x):
        # surface feature extraction
        x1 = self.relu(self.conv1(x))

        x2 = self.relu(self.conv2(x1))
        x31 = self.relu(self.conv31(x2))
        x32 = self.relu(self.conv32(x31))
        x3_f = self.sig(self.conv3f(x31))  # Pixel Attention 浅蓝色的线
        x33 = self.relu(self.conv33(x32))
        x3_out = self.lrelu(self.trans2(x33 * x3_f))  # 绿色模块转置卷积
        x3 = self.relu(self.conv3(x3_out))
        x2_f = self.sig(self.conv2f(x1))  # Pixel Attention 蓝色的线
        out = self.lrelu(self.trans1(x3 * x2_f))  # 紫色模块转置卷积
        return x + out # 红色的线

    def __init__(self):
        super(GridRefine, self).__init__()
        self.name = self.__class__.__name__
        f = 32
        self.conv1 = nn.Conv2d(2, f, kernel_size=3, stride=2, padding=1)
        self.conv_f =  nn.Conv2d(f,f,kernel_size=1)
        self.conv2 = nn.Conv2d(f,f, kernel_size=3, stride=2, padding=1)
        self.transConv1 = nn.ConvTranspose2d(f, f, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(f, 2, 2, 2, 0)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=True)
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        self.apply(_initialize_weights)

    def forward(self, x):
        c1_ = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1_))

        c3 = self.relu(self.transConv1(c2))
        cf = self.sigmoid(self.conv_f(c1_))
        c4 = self.transConv2(c3 * cf)
        m = self.lrelu(c4)
        return m + x


class GANet(nn.Module):
    def __init__(self, grid_shape=(5, 5), out_size=(1024, 1024), with_refine=True):
        super(GANet, self).__init__()
        self.grid_shape = grid_shape
        self.out_size = out_size
        self.with_refine = with_refine  # becomes WarpingNet w/o refine if set to false
        self.name = 'GANet' if not with_refine else 'GANet_without_refine'

        # relu
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))  # θaff ∈ R2×3

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2)
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape)#[5,5,2]
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(
            torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)  # θtps

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = GridRefine()
        else:
            self.grid_refine_net = None  # WarpingNet w/o refine

    # initialize WarpingNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat,
                                           torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute(
            (0, 3, 1, 2))  # grid generator φ(θaff)
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts,
                                               (1, x.size()[1]) + self.out_size)  # grid generator  φ(θtps) theta[1,27,2] ctrl_pts[5,5,2]
        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid)  # 第一个grid sampler ψ(gaff,gtps)

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(self.grid_refine_net(tps_grid), min=-1, max=1).permute(
                (0, 2, 3, 1))  # Refinement network
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):

        if self.fine_grid is None:
            # not simplified (training/validation)
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(self.affine_mat,
                                               torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute(
                (0, 3, 1, 2))
            coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(self.grid_refine_net(tps_grid), min=-1, max=1).permute((0, 2, 3, 1))
                fine_grid = fine_grid

            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)
        # warp
        x = F.grid_sample(x, fine_grid)  # 第二个 grid sampler
        return x


class CompenHD(nn.Module):
    def __init__(self, ga_net=None, pa_net=None):
        super(CompenHD, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        self.ga_net = copy.deepcopy(ga_net.module) if ga_net is not None else GANet()
        self.pa_net = copy.deepcopy(pa_net.module) if pa_net is not None else PANet()

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.ga_net.simplify(s)
        self.pa_net.simplify(self.ga_net(s))

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet (both x and s)
        x = self.ga_net(x)
        s = self.ga_net(s)
        # photometric compensation using CompenNet
        x = self.pa_net(x, s)
        return x


if __name__ == '__main__':
    input1 = torch.randn(4,3, 1024,1024).cuda()
    input2 = torch.randn(4, 3, 1024, 1024).cuda()
    net= CompenHD().cuda()
    a=net(input1,input2)
    print(a.size())
