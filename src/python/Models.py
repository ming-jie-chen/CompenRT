import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_tps
import math


class AttentionModel(nn.Module):
    def __init__(self,c1, k=3, s=1):
        super(AttentionModel, self).__init__()
        self.conv = nn.Conv2d(c1, 1, k, s, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv(x)
        attention_map = self.output_act(x1)
        output = x + x * torch.exp(attention_map)
        return output






#  PUNet at 256×256 resolution input
class PUNet256(nn.Module):
    def __init__(self):
        super(PUNet256, self).__init__()
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
        self.transConv1 = nn.ConvTranspose2d(128, 64,3,2,1,1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2,2,0)

        # s1
        self.skipConv11 = nn.Conv2d(3, 32, 3, 1, 1)
        self.skipConv12 = nn.Conv2d(32, 64,3,1,1)
        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1,1,0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        self.attention1 = AttentionModel(128)
        self.attention2 = AttentionModel(64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up_conv2 = nn.Conv2d(32, 3, 3, 1, 1)
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

    # x is the input uncompensated image, s is a surface image, the resolution is 256×256
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.skipConv12(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))


        res1 = self.relu(self.skipConv11(x))
        res1 = self.skipConv12(res1)

        res1 =res1-res1_s

        x = self.relu(self.conv1(x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 =res2-res2_s
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x - res3_s
        x=self.relu(self.conv4(x))

        x = self.relu(self.conv5(x))

        x = self.attention1(x)
        x=self.relu(self.transConv1(x)+res2)
        x = self.upsample1(x)
        x = self.relu(self.up_conv1(x) + res1)

        x=self.attention2(x)
        x=self.relu(self.transConv2(x))
        x = self.upsample2(x)
        x = self.up_conv2(x)

        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)
        return x

# PUNet at 512×512 resolution input
class PUNet512(nn.Module):
    def __init__(self):
        super(PUNet512, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.simplified = False
        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64,3,2,1,1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2,2,0)
        # output layer
        # s1
        self.skipConv11 = nn.Conv2d(3, 32, 3, 1, 1)
        self.skipConv12 = nn.Conv2d(32, 32,3,1,1)
        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1,1,0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        #s3
        self.skipConv31=nn.Conv2d(64,64,3,1,1)

        self.attention1 = AttentionModel(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up_conv2 = nn.Conv2d(32, 3, 3, 1, 1)
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
        res1_s = self.relu(self.skipConv12(s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        self.res3_s_pre = s
        s = self.relu(self.conv3(s))
        self.res4_s_pre = self.skipConv31(s)

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()
        self.res4_s_pre = self.res4_s_pre.squeeze()
        self.simplified = True

    # x is the input uncompensated image, s is a surface image, the resolution is 512×512
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
            res4_s = self.res4_s_pre
        else:

            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.skipConv12(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s=self.skipConv31(s)
            s = self.relu(self.conv3(s))
            res4_s=self.relu(self.conv4(s))

        res1 = self.relu(self.skipConv11(x))
        res1 = self.skipConv12(res1)
        res1 =res1-res1_s

        x = self.relu(self.conv1(x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 =res2-res2_s

        x = self.relu(self.conv2(x))

        res3=self.skipConv31(x)
        res3=res3-res3_s
        x = self.relu(self.conv3(x))

        x=self.relu(self.conv4(x))
        x = x - res4_s
        x = self.relu(self.conv5(x))

        x = self.attention1(x)
        x=self.relu(self.transConv1(x)+res3)
        x = self.upsample1(x)
        x = self.relu(self.up_conv1(x)+res2)

        x=self.relu(self.transConv2(x)+res1)
        x = self.upsample2(x)
        x = self.up_conv2(x)
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
        self.relu =  nn.ReLU()
        self.sig =  nn.Sigmoid()
        self.lrelu =  nn.LeakyReLU()

        self.norm1=nn.GroupNorm(num_groups=16,num_channels=32)
        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)
        self.apply(_initialize_weights)

    def forward(self, x):
        # surface feature extraction
        x1 = self.norm1(self.relu(self.conv1(x)))
        x2 = self.relu(self.conv2(x1))
        x31 = self.relu(self.conv31(x2))
        x32 = self.relu(self.conv32(x31))
        x3_f = self.sig(self.conv3f(x31))
        x33 = self.relu(self.conv33(x32))
        x3_out = self.lrelu(self.trans2(x33*x3_f))
        x3 = self.relu(self.conv3(x3_out))
        x2_f = self.sig(self.conv2f(x1))
        out = self.lrelu(self.trans1(x3*x2_f))
        return x+out


# The outsize parameter is used to adjust the size of the output resolution of the geometry correction.
class GDNet(nn.Module):
    def __init__(self, grid_shape=(5,5), out_size=(512,512), with_refine=True):
        super(GDNet, self).__init__()
        self.grid_shape = grid_shape
        self.out_size = out_size
        self.with_refine = with_refine  # becomes GDNet w/o refine if set to false
        self.name = 'GDNet' if not with_refine else 'GDNet_without_refine'

        # relu
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2)
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape)
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(
            torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = GridRefine()
        else:
            self.grid_refine_net = None  # GDNet w/o refine

    # initialize GDNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat,
                                           torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute(
            (0, 3, 1, 2))
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts,
                                               (1, x.size()[1]) + self.out_size)

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid)

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
        x = F.grid_sample(x, fine_grid)
        return x


class CompenRTFast(nn.Module):
    def __init__(self, gd_net=None, pu_net=None):
        super(CompenRTFast, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        self.gd_net = copy.deepcopy(gd_net.module) if gd_net is not None else GDNet()
        self.pu_net = copy.deepcopy(pu_net.module) if pu_net is not None else PUNet256()
    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.gd_net.simplify(s)
        self.pu_net.simplify(self.gd_net(s))

    def forward(self, x, s):
        # geometric correction using GDNet (both x and s)
        x = self.gd_net(x)
        s = self.gd_net(s)
        # x and s is Bx3x256x256 warped image
        # photometric compensation using PUNet
        x= self.pu_net(x,s)
        return x


class CompenRT(nn.Module):
    def __init__(self, gd_net=None, pu_net=None):
        super(CompenRT, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        self.gd_net = copy.deepcopy(gd_net.module) if gd_net is not None else GDNet()
        self.pu_net = copy.deepcopy(pu_net.module) if pu_net is not None else PUNet512()
    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.gd_net.simplify(s)
        self.pu_net.simplify(self.gd_net(s))

    def forward(self, x, s):
        # geometric correction using GDNet (both x and s)
        x = self.gd_net(x)
        s = self.gd_net(s)
        # x and s is Bx3x512x512 warped image
        # photometric compensation using PUNet
        x= self.pu_net(x,s)
        return x

if __name__ == '__main__':
    from profile1 import *
    import os
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if torch.cuda.device_count() >= 1:
        print('Train with', len(device_ids), 'GPUs!')
    else:
        print('Train with CPU!')
    input1 = torch.randn(1,3, 1024,1024).to(device)
    compen_rt =CompenRT().to(device)
    a, b = profile(compen_rt,input1)
    print(a)
    print(b)

