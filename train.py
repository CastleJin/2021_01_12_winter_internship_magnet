import torch
import numpy as np

from tqdm import tqdm
from matplotlib.pyplot import imshow
from torchvision import transforms
from module import magnet
from dataloader import read_and_decode_3frame
from utils import ToTensor, shot_noise, num_sampler, save_image

# inference
# temporal_filter
# interface

class training(object):
    def __init__(self):
        # PATH
        self.PARA_PATH = '/home/urp1/model'
        self.DATA_PATH = 'home/urp1/train/data/train'
        self.load_num = 100
        self.load_name = '/epoch_{}'.format(self.load_num)

        # preprocessing
        self.poisson_noise_n = 0.3

        # data size
        self.train_size = 90000
        self.val_size = 10000
        self.train_batch_size = 1
        self.val_batch_size = 1

        # for exponential decay
        self.decay_steps = 3000
        self.lr_decay = 1.0

        # for optimizer
        self.betal = 0.9

        # for training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_load = False
        self.num_epoch = 400
        self.remain_epoch = self.num_epoch
        self.tex_loss_w = 1.0
        self.sha_loss_w = 1.0

        # loss
        self.train_losses = []
        self.val_losses = []

    def _load(self):
        self.model = magnet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001, betas = (self.betal, 0.999), weight_decay = 0, amsgrad=False)
        self.criterion = torch.nn.L1Loss()

        if self.is_load:            
            checkpoint = torch.load(self.PARA_PATH+self.load_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.remain_epoch -= checkpoint['epoch']
            self.train_losses = checkpoint['train_loss'][:]
            self.val_losses = checkpoint['val_loss'][:]

    def _forward_backward_propagation(self, train_loader, val_loader, is_train=True):
        running_loss = 0.0
        loader = (train_loader if is_train else val_loader)
        
        for _, sample in enumerate(loader):
                amplified, frameA, frameB, frameC, amp_factor = sample['amplified'].to(self.device), \
                                                                sample['frameA'].to(self.device), \
                                                                sample['frameB'].to(self.device), \
                                                                sample['frameC'].to(self.device), \
                                                                sample['mag_factor'].to(self.device)
                if is_train: 
                    self.optimizer.zero_grad()
                
                Y, Va, Vb, _, _, Mb, Mb_ = self.model(amplified, frameA, frameB, frameC, amp_factor)
                loss = self.criterion(Y, amplified) + self.tex_loss_w * self.criterion(Va, Vb) + self.sha_loss_w * self.criterion(Mb, Mb_)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()

        return running_loss
    
    def get_data(self):
        train_dataset = read_and_decode_3frame(data_path = self.DATA_PATH, 
                                               transform = transforms.Compose([ToTensor(), shot_noise(self.poisson_noise_n)]))
        val_dataset = read_and_decode_3frame(data_path = self.DATA_PATH, 
                                             transform = transforms.Compose([ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=self.train_batch_size, 
                                                   sampler=num_sampler(train_dataset, 
                                                   is_val=False, 
                                                   shuffle=True))
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=self.val_batch_size, 
                                                 sampler=num_sampler(val_dataset))
        return train_dataset, val_dataset, train_loader, val_loader

    def test(self):
        train_dataset, _,_,_ = self.get_data()
        # test one image
        i = 0
        sample = train_dataset[i]
        print(i, sample['amplified'].shape, sample['mag_factor'], sample['mag_factor'].shape)
        a = sample['frameA']
        print(a)
        b= transforms.ToPILImage()(a)
        save_image(b, '/home/urp1/result.png')
        
    def train(self):
        self._load()
        _, _, train_loader, val_loader = self.get_data()

        for epoch in tqdm(range(self.remain_epoch)):
            #train
            self.model.train()
            train_loss = self._forward_backward_propagation(train_loader, val_loader, is_train=True)

            # evaluation
            self.model.eval()
            with torch.no_grad():
                val_loss = self._forward_backward_propagation(train_loader, val_loader, is_train=False)

            cal_train_loss = train_loss / self.train_size
            cal_val_loss = val_loss / self.val_size
            
            # result
            print('[epoch: %d] train_loss: %.3f, val_loss: %.3f' % (epoch + 1, cal_train_loss, cal_val_loss))
            self.train_losses.append(cal_train_loss)
            self.val_losses.append(cal_val_loss)

            # save
            torch.save({'epoch': epoch+1, 
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(), 
                        'train_loss': self.train_losses, 
                        'val_loss': self.val_losses}, 
                        self.PARA_PATH+'/epoch_{}'.format(epoch+1))

            # for re-train
            torch.cuda.empty_cache()