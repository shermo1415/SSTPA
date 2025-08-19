import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def makesampleindex(df, interval=6, outer = 6, gap=3):
    xindlist = []
    yindlist = []
    for i in range(0, len(df['day'])-interval-outer-2, gap):
        d0 = datetime.strptime(
            str(df['year'][i]) + '-' + str(df['month'][i]) + '-' + str(df['day'][i]) + '-' + str(df['hour'][i]) + '-' + str(
                df['minute'][i]), '%Y-%m-%d-%H-%M')
        d1 = datetime.strptime(
            str(df['year'][i+interval+1]) + '-' + str(df['month'][i+interval+1]) + '-' + str(df['day'][i+interval+1]) + '-' + str(df['hour'][i+interval+1]) + '-' + str(
                df['minute'][i+interval+1]), '%Y-%m-%d-%H-%M')

        d2 = datetime.strptime(
            str(df['year'][i+interval+outer+1]) + '-' + str(df['month'][i+interval+outer+1]) + '-' + str(df['day'][i+interval+outer+1]) + '-' + str(df['hour'][i+interval+outer+1]) + '-' + str(
                df['minute'][i+interval+outer+1]), '%Y-%m-%d-%H-%M')

        if interval<23 and outer <24:
            if (d1 - d0).seconds == (interval + 1) * 3600:
                if (d2 - d1).seconds == (outer) * 3600:
                    xtoappend = [i]
                    for int in range(1, interval + 1):
                        xtoappend.append(i+int)
    
                    ytoappend = [interval + 1+i]
                    for int in range(interval + 2, interval +outer + 1):
                        ytoappend.append(i+int)
                    xindlist.append(xtoappend)
                    yindlist.append(ytoappend)
        else:
            if (d1-d0).days == 1 and (d1 - d0).seconds == 0:
                if (d2-d1).days == 1 and (d2 - d1).seconds == 0:
                    xtoappend = [i]
                    for int in range(1, interval + 1):
                        xtoappend.append(i+int)
    
                    ytoappend = [interval + 1+i]
                    for int in range(interval + 2, interval +outer + 1):
                        ytoappend.append(i+int)
                    xindlist.append(xtoappend)
                    yindlist.append(ytoappend)
    return xindlist, yindlist

class TimeDownScalingDataset(Dataset):
    def __init__(self, dataset_path = '24composite.npy', dims =['t2m', 'sp', 'u10'], interval = 6, outer = 6, gap = 3, fy=False):
        raw_data = np.load(dataset_path, allow_pickle=True).item()
        self.xindl, self.yindl = makesampleindex(raw_data, interval, outer, gap)
        self.meta_data = []
        for dim in dims:
            self.meta_data.append(raw_data[dim])
        self.meta_data = np.stack(self.meta_data, 1)
        # if fy:
        #     self.meta_fy4b = raw_data['fy']
        self.meta_time = np.stack([raw_data['year'], raw_data['month'], raw_data['day'], raw_data['hour'], raw_data['minute']],1)
        self.interval = interval
        self.fy = fy
    def __len__(self):
        return len(self.yindl)

    def GetDataShape(self):
        return self.meta_data.shape, self.meta_time.shape, len(self.yindl)

    def __getitem__(self, index):
        xind = self.xindl[index]
        yind = self.yindl[index]

        img01 = self.meta_data[xind]
        
        imggt = self.meta_data[yind]
        time = self.meta_time[xind]

        # fy01 = self.meta_fy4b[xind]
        
        

        return img01, imggt, time, time#, fy01#, fygt

if __name__ == "__main__":

    dataset_path = r'E:\PycharmProjectsPy312\9\Dataset\Dataset1\24composite_onezero_test.npy'
    # dims = ['t2m', 'sp', 'u10']
    # inter_frames = 2
    # outer_frames = 2
    # 
    # d = TimeDownScalingDataset(dataset_path, dims, inter_frames, outer_frames)
    # train_data = DataLoader(d, batch_size=4, pin_memory=True, drop_last=True)
    # 
    # for i, k in enumerate(train_data):
    #     print(k[4], k[5])
    #
    # np.save('im01.npy', k[0][0])
    # np.save('imgt.npy', k[1][0])
    # np.save('fy01.npy', k[2][0])
    # np.save('fygt.npy', k[3][0])