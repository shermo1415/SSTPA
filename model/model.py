from .FrameExtraction import FrameExtraction
from .CrossFrameAttention import CrossFrameAttentionSubNet
from .FlowEstimate import FlowEst
import torch
import torch.nn as nn
import math
        
class network(nn.Module):
    def __init__(self, Cin, emb_dim, blocks, attn_layer):
        super().__init__()
        self.FE_wd = FrameExtraction(Cin, emb_dim*4)
        self.FE_mt = FrameExtraction(Cin, emb_dim*4)
        # self.FE_fy = FrameExtraction(6, emb_dim*4)
        self.CFA = CrossFrameAttentionSubNet(blocks, attn_layer, emb_dim * 4, emb_dim * 8, (16, 16), 8)
        self.ES = FlowEst(blocks,2,emb_dim*4)
    def forward(self, wd, mt, fy, time, in_frames, out_frames, train):
        f1_wd, f2_wd = self.FE_wd(wd)
        # f1_mt, f2_mt = self.FE_mt(mt)
        # f1_fy, f2_fy = self.FE_fy(fy)
        
        f_fb, l_fb = self.CFA(f1_wd, f2_wd, time, f1_wd, f2_wd, f1_wd, f2_wd)
        
        out = self.ES(wd[:,0], wd[:,-1], f_fb, l_fb, in_frames, out_frames, train)
        return out