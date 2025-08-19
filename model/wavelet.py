import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import cv2

def upsample_coeff(coeff, target_shape):
    return cv2.resize(coeff, target_shape, interpolation=cv2.INTER_LINEAR)

def SingleChannelwavelet2d(x, wavelet, level):
    target_shape = x.shape
    coeffs = pywt.wavedec2(x.cpu().detach(), wavelet, level=level, mode='symmetric')
    approx = upsample_coeff(coeffs[0], target_shape)
    dcoeffs = [upsample_coeff(d, target_shape) for d_list in coeffs[1:] for d in d_list]
    return np.concatenate([np.stack(dcoeffs, axis=0), approx[np.newaxis,:]],0)

def MultiChannelwavelet2d(x, cs, wavelet, level):
    '''X:  C,H,W'''
    cf = []
    for c in range(cs):
        cf.append(SingleChannelwavelet2d(x[c], wavelet, level))
    return np.concatenate(cf,0)

def wavelet3d(x, cs, wavelet, level):
    '''X:  T,C,H,W'''
    ts = []
    for t in range(x.shape[0]):
        ts.append(MultiChannelwavelet2d(x[t], cs, wavelet, level))
    return np.stack(ts)

def Batchwavelet3d(x, cs, wavelet, level):
    '''X:  B,T,C,H,W'''
    bs = []
    for b in range(x.shape[0]):
        bs.append(wavelet3d(x[b], cs, wavelet, level))
    return np.stack(bs).swapaxes(1,2)

class SpatialWaveletExtractor(nn.Module):
    def __init__(self, Cin, embbedding):
        super(SpatialWaveletExtractor, self).__init__()
        self.branch = nn.Sequential(nn.Conv3d(in_channels=Cin*10, out_channels=embbedding, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(embbedding),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels=embbedding, out_channels=embbedding, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(embbedding),
                    nn.ReLU(inplace=True))
        self.Cin = Cin

    def forward(self, x):
        x = Batchwavelet3d(x, self.Cin, 'sym4', 3)
        x = self.branch(torch.tensor(x).float().cuda())
        return x