import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model.model import network
from makedataset import TimeDownScalingDataset
import pickle
import math
from torch.cuda.amp import autocast, GradScaler
from config import configs
from log import printwrite
file = "log/log.txt"

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.Cin = configs.Cin
        self.network =network(configs.Cin, configs.emb_dim, configs.blocks, configs.attn_layer).to(self.device)
        # self.awl = AutomaticWeightedLoss(configs.Cin)
        
        self.opt = torch.optim.Adam(self.network.parameters(), lr = configs.opt)#, weight_decay=configs.weight_decay)
        self.scaler = GradScaler()     

    def test_loss(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)
    
    def test(self, dataloader_eval):
        self.network.eval()
        predlist = []
        truelist = []
        with torch.no_grad():
            for img, imggt, time, fy in dataloader_eval:
                
                wd, mt = img[:,:,:2], img[:,:,2:]
                pred = self.network(wd.float().to(self.device), mt.float().to(self.device), fy.float().to(self.device), time, configs.interinterval+1, configs.outerinterval, False)
                predlist.append(pred)
                truelist.append(imggt[:,:,:2].float().to(self.device))
                
            predlist =torch.cat(predlist, dim=0)
            truelist =torch.cat(truelist, dim=0)

            testloss = self.test_loss(predlist, truelist)
            np.save("result/pred.npy", predlist.cpu().detach().numpy())
            np.save("result/true.npy", truelist.cpu().detach().numpy())
        return testloss

########################################################################################################################
if __name__ == '__main__':
    
    # dataset_train = TimeDownScalingDataset(configs.train_path, configs.dims, configs.interinterval, configs.outerinterval, configs.samplegap)
    # # dataset_train.indexs = dataset_train.indexs[::25]
    # print(dataset_train.GetDataShape())

    dataset_eval = TimeDownScalingDataset(configs.val_path, configs.dims, configs.interinterval, configs.outerinterval, configs.samplegap, True)
    # dataset_eval.indexs = dataset_eval.indexs[::25]
    print(dataset_eval.GetDataShape())
    dataloader_eval = DataLoader(dataset_eval, batch_size=configs.batch_size_test, shuffle=False)
    trainer = Trainer(configs)
    
    trainer.network = torch.load('exp/SRNet.pt')
    # model = torch.load('exp/SRNet.pt')
    # trainer.network.load_state_dict(model['net'])
    # trainer.network.eval()
    print(trainer.test(dataloader_eval))