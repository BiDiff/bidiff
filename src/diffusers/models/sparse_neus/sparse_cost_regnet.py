import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor

# from tsparse.torchsparse_utils import *
# from .torchsparse_utils import *
from diffusers.models.sparse_neus.torchsparse_utils import *


# __all__ = ['SPVCNN', 'SConv3d', 'SparseConvGRU']


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """

    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1),
            ConvBnReLU(8, 8, 3, 1, 1))

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2),
            ConvBnReLU(16, 16, 3, 1, 1),
            ConvBnReLU(16, 16, 3, 1, 1))

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2),
            ConvBnReLU(32, 32, 3, 1, 1),
            ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return torch.nn.functional.interpolate(x, scale_factor=2,
                                               mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        return [feat2, feat1, feat0]  # coarser to finer features


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SparseResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), 
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), 
            spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicSparseConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            SparseResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            SparseResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicSparseConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            SparseResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            SparseResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicSparseDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                SparseResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                                    dilation=1),
                SparseResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicSparseDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                SparseResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                                    dilation=1),
                SparseResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1)
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F)

        return z3.F


class SparseCostRegNet(nn.Module):
    """
    Sparse cost regularization network;
    require sparse tensors as input
    """

    def __init__(self, d_in, d_out=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

    def forward(self, x, emb=None):
        """

        :param x: sparse tensor
        :return: sparse tensor
        """
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F

class SparseCostRegNetV2(nn.Module):
    """
    Sparse cost regularization network;
    require sparse tensors as input
    """

    def __init__(self, d_in, d_out=8, temb_channels=320):
        super(SparseCostRegNetV2, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nonlinearity = nn.GELU()

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)
        # self.time_emb_proj0 = torch.nn.Linear(temb_channels, 2 * d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)
        self.conv2_ = BasicSparseConvolutionBlock(16, 16)
        self.time_emb_proj2 = torch.nn.Linear(temb_channels, 2 * 16)
        self.norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=16, eps=1e-6, affine=True)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)
        self.conv4_ = BasicSparseConvolutionBlock(32, 32)
        self.time_emb_proj4 = torch.nn.Linear(temb_channels, 2 * 32)
        self.norm4 = torch.nn.GroupNorm(num_groups=16, num_channels=32, eps=1e-6, affine=True)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)
        self.conv6_ = BasicSparseConvolutionBlock(64, 64)
        self.time_emb_proj6 = torch.nn.Linear(temb_channels, 2 * 64)
        self.norm6 = torch.nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6, affine=True)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)
    
    def forward_resblock(self, x, conv1, conv2, time_emb_proj, norm, emb=None):
        input_tensor = x.F
        hidden_states = input_tensor
        hidden_states = norm(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        x.F = hidden_states
        x = conv1(x)
        # NOTE(lihe): add gelu
        emb = self.nonlinearity(emb)
        temb = time_emb_proj(emb)
        scale, shift = torch.chunk(temb, 2, dim=1)
        x.F = x.F * (1 + scale) + shift
        x.F = self.nonlinearity(x.F)
        x = conv2(x)
        x.F = x.F + input_tensor # residual
        return x

    def forward(self, x, emb=None):
        """

        :param x: sparse tensor
        emb: b, c
        :return: sparse tensor
        """
        bs = emb.shape[0]
        assert bs == 1, "currently we only support bs = 1"
        conv0 = self.conv0(x)
        
        conv1 = self.conv1(conv0)
        conv2 = self.forward_resblock(conv1, self.conv2, self.conv2_, 
                                      self.time_emb_proj2, self.norm2,
                                      emb=emb)

        # conv4 = self.conv4(self.conv3(conv2))
        conv3 = self.conv3(conv2)
        conv4 = self.forward_resblock(conv3, self.conv4, self.conv4_, 
                                      self.time_emb_proj4, self.norm4,
                                      emb=emb)

        # x = self.conv6(self.conv5(conv4))
        conv5 = self.conv5(conv4)
        x = self.forward_resblock(conv5, self.conv6, self.conv6_, 
                                      self.time_emb_proj6, self.norm6,
                                      emb=emb)

        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F

class SparseCostRegNetV3(nn.Module):
    """
    Sparse cost regularization network;
    require sparse tensors as input
    """

    def __init__(self, d_in, d_out=8, temb_channels=320):
        super(SparseCostRegNetV2, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nonlinearity = nn.GELU()

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)
        # self.time_emb_proj0 = torch.nn.Linear(temb_channels, 2 * d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)
        self.conv2_ = BasicSparseConvolutionBlock(16, 16)
        self.time_emb_proj2 = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, 2 * 16)
        )
        self.norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=16, eps=1e-6, affine=True)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)
        self.conv4_ = BasicSparseConvolutionBlock(32, 32)
        self.time_emb_proj4 = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, 2 * 32)
        )
        self.norm4 = torch.nn.GroupNorm(num_groups=16, num_channels=32, eps=1e-6, affine=True)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)
        self.conv6_ = BasicSparseConvolutionBlock(64, 64)
        self.time_emb_proj6 = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, 2 * 64)
        )
        self.norm6 = torch.nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6, affine=True)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)
    
    def forward_resblock(self, x, conv1, conv2, time_emb_proj, norm, emb=None):
        input_tensor = x.F.clone()
        x.F = norm(x.F)
        temb = time_emb_proj(emb)
        scale, shift = torch.chunk(temb, 2, dim=1)
        x.F = x.F * (1 + scale) + shift
        x = conv1(x)
        x.F = self.nonlinearity(x.F)
        x = conv2(x)
        x.F = x.F + input_tensor # residual
        return x

    def forward(self, x, emb=None):
        """

        :param x: sparse tensor
        emb: b, c
        :return: sparse tensor
        """
        bs = emb.shape[0]
        assert bs == 1, "currently we only support bs = 1"
        conv0 = self.conv0(x)
        
        conv1 = self.conv1(conv0)
        conv2 = self.forward_resblock(conv1, self.conv2, self.conv2_, 
                                      self.time_emb_proj2, self.norm2,
                                      emb=emb)

        # conv4 = self.conv4(self.conv3(conv2))
        conv3 = self.conv3(conv2)
        conv4 = self.forward_resblock(conv3, self.conv4, self.conv4_, 
                                      self.time_emb_proj4, self.norm4,
                                      emb=emb)

        # x = self.conv6(self.conv5(conv4))
        conv5 = self.conv5(conv4)
        x = self.forward_resblock(conv5, self.conv6, self.conv6_, 
                                      self.time_emb_proj6, self.norm6,
                                      emb=emb)

        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F

class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)
        out.F = out.F + self.point_transforms(z.F)
        return out
