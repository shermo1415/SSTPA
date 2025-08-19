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
def SSIM_loss(pred, true):
    pred_np = pred[:,:,:2].permute(1,0,2,3,4)
    true_np = true[:,:,:2].permute(1,0,2,3,4)
    total_loss = 0.0

    for i in range(pred_np.shape[0]):
              loss = 1 - SSIM.SSIM(pred_np[i], true_np[i])
              total_loss += loss.item()

    average_loss = total_loss / (pred_np.shape[0])
    return average_loss

def Angle_loss(batch_y, pred_y):
    true = Angle_wind(batch_y)
    pred = Angle_wind(pred_y)
    
    diff = torch.abs(true - pred)
    diff = torch.where(diff > 180, 360 - diff, diff)

    diff_normalized = diff / 180.0

    mse = torch.mean(diff_normalized ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def Angle_wind(y):
    a_fushu = y[:, :, 0, :, :]
    b_fushu = y[:, :, 1, :, :]
    complex_tensor = a_fushu + 1j * b_fushu
    angle_rad = torch.angle(complex_tensor)
    angle_deg = angle_rad * (180 / 3.141592653589793)
    angle_metric = angle_deg.unsqueeze(2)
    return angle_metric

def EngeryLoss(y_trues, y_preds):
    fs = y_preds.shape[1]
    for f in range(fs-1):
        TrueDeltaEn = (y_trues[0,f+1] - y_trues[0,f]) ** 2
        PredDeltaEn = (y_preds[0,f+1] - y_preds[0,f]) ** 2
    return F.l1_loss(TrueDeltaEn, PredDeltaEn)
    
class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.Cin = configs.Cin
        self.network =network(configs.Cin, configs.emb_dim, configs.blocks, configs.attn_layer).to(self.device)
        # self.awl = AutomaticWeightedLoss(configs.Cin)
        
        self.opt = torch.optim.Adam(self.network.parameters(), lr = configs.opt)#, weight_decay=configs.weight_decay)
        self.scaler = GradScaler()
        self.interinterval = configs.interinterval
        
    def train_loss(self, y_preds, y_trues, i0, i1):
        return F.l1_loss(y_preds, y_trues) + Angle_loss(y_trues, y_preds) + EngeryLoss(y_trues, y_preds)

    def test_loss(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)
    
    def train_once(self, img, imggt, fy, time):
        img = img.float().to(self.device)
        i1 = img[:,0]
        i2 = img[:,-1]
        imggt = imggt.float().to(self.device)
        wd, mt = img[:,:,:2], img[:,:,:2]
        pred = self.network(wd.float().to(self.device), mt.float().to(self.device), fy.float().to(self.device), time, configs.interinterval+1, configs.outerinterval, True)
        loss = self.train_loss(pred, imggt[:,:,:2], i1, i2)
        return loss
    
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
        return testloss
    
    def train(self, dataset_train, dataset_eval, chk_path):
        # torch.manual_seed(0)
        printwrite(file, 'loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        printwrite(file, 'loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)
        count = 0
        best = math.inf
        self.network.train()
        for i in range(self.configs.num_epochs):
            printwrite(file, '\nepoch: {0}'.format(i + 1))
            j = 0         
            data_iter = iter(dataloader_train)
            self.opt.zero_grad()
            while j < len(dataloader_train):
                j += 1
                self.network.train()

                with autocast():
                    img, imggt, time, fy = next(data_iter)   
                    loss = self.train_once(img, imggt, fy, time)

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                if (j + 1) % self.configs.display_interval == 0:
                    printwrite(file, 'batch training loss: {:.4f}'.format(loss))
                if (j + 1) % (self.configs.display_interval * configs.eval_interval) == 0:
                    loss = self.test(dataloader_eval)
                    
                    printwrite(file, 'batch eval loss: {:.4f}'.format(loss))
                    if loss < best:
                        count = 0
                        printwrite(file, 'eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss))           
                        self.save_model(chk_path)
                        best = loss
                        
            loss = self.test(dataloader_eval)
            printwrite(file, 'epoch eval loss: {:.4f}'.format(loss))
            if loss >= best:
                count += 1
                printwrite(file, 'eval loss is not reduced for {} epoch'.format(count))
                printwrite(file, 'best is {} until now'.format(best))
            else:
                count = 0
                printwrite(file, 'eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss))
                self.save_model(chk_path)
                best = loss
            self.save_model('exp/last.chk')
    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)
    def save_model(self, path):
        torch.save(self.network, path)
        # torch.save({'net': self.network.state_dict()}, path)
########################################################################################################################
if __name__ == '__main__':
    
    dataset_train = TimeDownScalingDataset(configs.train_path, configs.dims, configs.interinterval, configs.outerinterval, configs.samplegap, False)
    # dataset_train.xindl, dataset_train.yindl = dataset_train.xindl[::25], dataset_train.yindl[::25]
    print(dataset_train.GetDataShape())

    dataset_eval = TimeDownScalingDataset(configs.val_path, configs.dims, configs.interinterval, configs.outerinterval, configs.samplegap, False)
    dataset_eval.xindl, dataset_eval.yindl = dataset_eval.xindl[::25], dataset_eval.yindl[::25]
    print(dataset_eval.GetDataShape())
    trainer = Trainer(configs)
    trainer.save_configs('exp/config_train.pkl')

    # model = torch.load('exp/SRNet_200e.chk')
    # trainer.network.load_state_dict(model['net'])
    
    trainer.train(dataset_train, dataset_eval, 'exp/SRNet.pt')