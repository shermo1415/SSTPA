import torch

class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.device = torch.device('cuda:0')

configs.batch_size = 8#8#4倍4； 8倍32； #16
configs.batch_size_test = 8

configs.num_epochs = 100
configs.opt = 0.0001 #原 0.001； swin 0.0002

configs.train_path = r'/root/autodl-tmp/D2_212223_onezero.npy'
configs.val_path = r'/root/autodl-tmp/D2_24_onezero.npy'
configs.dims = ['u100', 'v100']
configs.interinterval = 23
configs.outerinterval = 24
configs.trainframes = configs.outerinterval
configs.samplegap = 3

# configs.image_size = (64, 64)
configs.train_shuffle = False

configs.Cin = 2
configs.emb_dim = 64 #128
configs.attn_layer = 2 #4
configs.blocks = 2

configs.display_interval = 110
configs.eval_interval = 5