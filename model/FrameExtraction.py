from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import SpatialWaveletExtractor

class Complexconv3x3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Complexconv3x3x3, self).__init__()
        self.Rconv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.Iconv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    def forward(self, xr, xi):
        yr = self.Rconv(xr) - self.Iconv(xi)
        yi = self.Iconv(xr) + self.Rconv(xi)
        return yr, yi

class Complexconv1x1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Complexconv1x1x1, self).__init__()
        self.Rconv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.Iconv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
    def forward(self, xr, xi):
        yr = self.Rconv(xr) - self.Iconv(xi)
        yi = self.Iconv(xr) + self.Rconv(xi)
        return yr, yi

class ComplexBatchNorm3d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm3d, self).__init__()
        self.Rb = nn.BatchNorm3d(num_features)
        self.Ib = nn.BatchNorm3d(num_features)
    def forward(self, xr, xi):
        return self.Rb(xr), self.Ib(xi)

class ComplexReLU(nn.Module):
    def __init__(self, inplace=True):
        super(ComplexReLU, self).__init__()
        self.Rrelu = nn.ReLU(inplace)
        self.Irelu = nn.ReLU(inplace)
    def forward(self, xr, xi):
        return self.Rrelu(xr), self.Irelu(xi)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = ComplexBatchNorm3d
        bottleneck_planes = groups * planes
        self.conv1 = Complexconv1x1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([Complexconv3x3x3(bottleneck_planes // scales, bottleneck_planes // scales) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = Complexconv1x1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ComplexReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsamplec =  Complexconv1x1x1(inplanes, planes * self.expansion, stride)
        self.downsampleb = norm_layer(planes * self.expansion)

        self.stride = stride
        self.scales = scales

    def forward(self, x):
        xr, xi = x[0], x[1]
        identityr, identityi = xr, xi

        outr, outi = self.conv1(xr, xi)
        outr, outi = self.bn1(outr, outi)
        outr, outi = self.relu(outr, outi)
        xsr = torch.chunk(outr, self.scales, 1)
        xsi = torch.chunk(outi, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append([xsr[s], xsi[s]])
            elif s == 1:
                ir, ii = self.conv2[s-1](xsr[s], xsi[s])
                ir, ii = self.bn2[s-1](ir, ii)
                ys.append(self.relu(ir, ii))
            else:
                ir, ii = self.conv2[s - 1](xsr[s] + ys[-1][0], xsi[s] + ys[-1][1])
                ir, ii = self.bn2[s-1](ir, ii)
                ys.append(self.relu(ir, ii))
        outr = torch.cat([ys[0][0], ys[1][0], ys[2][0], ys[3][0]], 1)
        outi = torch.cat([ys[0][1], ys[1][1], ys[2][1], ys[3][1]], 1)

        outr, outi = self.conv3(outr, outi)
        outr, outi = self.bn3(outr, outi)

        # if self.se is not None:
        #     out = self.se(out)

        if self.downsamplec is not None:
            identityr, identityi = self.downsamplec(identityr, identityi)
            identityr, identityi = self.downsampleb(identityr, identityi)
        outr += identityr
        outi += identityi
        outr, outi = self.relu(outr, outi)

        return outr, outi

def _make_layer(block, Cin, planes, blocks, stride = 2):
    if stride:
        downsample = nn.Sequential(
            Complexconv3x3x3(Cin, planes * block.expansion, stride),
            ComplexBatchNorm3d(planes * block.expansion))

    layers = []
    layers.append(
        block(inplanes=Cin,
              planes=planes,
              stride=stride,
              downsample=None))
    in_planes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_planes, planes))
    return nn.Sequential(*layers)

class emb(nn.Module):
    def __init__(self,Cin, planes):
        super().__init__()
        self.layer = _make_layer(Res2NetBottleneck, Cin, planes, 4, 2)

    def forward(self, x):
        return self.layer(x)

class timemodule(nn.Module):
    def __init__(self, cin, dt, kernelsize=3):
        super().__init__()
        if kernelsize == 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(cin*dt, cin*dt // 4, 3, 1, 1),
                nn.BatchNorm2d(cin*dt // 4),
                nn.ReLU(),
                nn.Conv2d(cin*dt // 4, 2*cin, 3, 1, 1),
                nn.BatchNorm2d(2*cin)
            )
        elif kernelsize == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(cin*dt, cin*dt // 4, 1, 1, 0),
                nn.BatchNorm2d(cin*dt // 4),
                nn.ReLU(),
                nn.Conv2d(cin*dt // 4, 2*cin, 1, 1, 0),
                nn.BatchNorm2d(2*cin)
            )

    def forward(self, x):
        B, C, T, W, H = x.shape
        o = x.view(B, C * T, W, H)
        o = self.conv1(o)
        o = o.view(B, 2*C, W, H)
        return o #+ res

class FrameExtraction(nn.Module):
    def __init__(self, Cin, emb_dim):
        super().__init__()
        self.stwavelayer = SpatialWaveletExtractor(Cin, emb_dim//2)
        self.conv = nn.Sequential(nn.Conv3d(in_channels=Cin, out_channels=emb_dim//2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(emb_dim//2),
                    nn.ReLU(inplace=True))
        self.embn = emb(emb_dim//2, emb_dim//2)
        self.timemodule = timemodule(emb_dim, 12, kernelsize=3)
        self.emb_dim = emb_dim

    def forward(self, x):
        xw = self.stwavelayer(x)
        xd = self.conv(x.transpose(1,2))

        xr, xi = self.embn([xw, xd])
        x = torch.cat([xr, xi], dim=1)
        x = self.timemodule(x)
        return x[:,:self.emb_dim], x[:,self.emb_dim:]

