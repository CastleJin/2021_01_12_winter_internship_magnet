# most codes from https://people.csail.mit.edu/tiam/deepmag/
# Refereces
# 1. https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686

from __future__ import print_function

import os
import torch
import time
import math
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from glob import glob
from module import magnet
from torchvision import transforms
from dataloader import ImageFolderLMDB
from scipy.signal import firwin, butter
from utils import *

DEFAULT_VIDEO_CONVERTER = 'ffmpeg'

class mag(object):
    def __init__(self, checkpoint_file_path):
        # PATH
        self.PARA_PATH = os.path.dirname(checkpoint_file_path)
        self.load_name = os.path.basename(checkpoint_file_path)

        # preprocessing
        self.poisson_noise_n = 0.3

        # data size
        self.total_image_size = 100000
        self.val_num = self.total_image_size / 10
        self.train_num = self.total_image_size - self.val_num
        
        # iter size
        self.train_batch_size = 4
        self.val_batch_size = 4
        self.train_size = math.ceil(self.train_num / self.train_batch_size)
        self.val_size = math.ceil(self.val_num / self.val_batch_size) 

        # for exponential decay
        self.decay_steps = 3000
        self.lr_decay = 1.0

        # for optimizer
        self.betal = 0.9

        # for training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_load = True
        self.num_epoch = 100
        self.remain_epoch = self.num_epoch
        self.tex_loss_w = 1.0
        self.sha_loss_w = 1.0

        # loss
        self.train_losses = []
        self.val_losses = []

    def _load(self):
        self.model = magnet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001, betas = (self.betal, 0.999), weight_decay = 0, amsgrad=False)
        self.criterion = torch.nn.L1Loss()

        if self.is_load:            
            checkpoint = torch.load(os.path.join(self.PARA_PATH, self.load_name), 
                                    map_location = self.device)
            state_dict = checkpoint['model_state_dict']
            # Note Reference 1.
            # remove 78 ~ 82 lines for loading single-gpu trained model
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name]=v

            self.model.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_loss'][:]
            self.val_losses = checkpoint['val_loss'][:]
            self.remain_epoch = self.num_epoch - len(self.train_losses)

    def get_data(self, datapath):
        train_dataset = ImageFolderLMDB(datapath, 
                                        transform = transforms.Compose([ToTensor(), shot_noise(self.poisson_noise_n)]))
        val_dataset = ImageFolderLMDB(datapath, 
                                        transform = transforms.Compose([ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=self.train_batch_size, 
                                                   num_workers = 20,
                                                   sampler=num_sampler(train_dataset, 
                                                   is_val=False, 
                                                   shuffle=True))
        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=self.val_batch_size,
                                                 num_workers = 20,
                                                 sampler=num_sampler(val_dataset))
        return train_dataset, val_dataset, train_loader, val_loader

    def print_loss(self):
        self._load()
        iters = range(0, len(self.train_losses))
        plt.plot(iters, self.train_losses, color='dodgerblue', label = 'Training loss')
        plt.plot(iters, self.val_losses, color='sandybrown', label = 'Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('iters')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./Loss_graph.png')

    def _get_val_loss(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _, sample in enumerate(val_loader):
                    amplified, frameA, frameB, frameC, amp_factor = sample['amplified'].to(self.device), \
                                                                    sample['frameA'].to(self.device), \
                                                                    sample['frameB'].to(self.device), \
                                                                    sample['frameC'].to(self.device), \
                                                                    sample['mag_factor'].to(self.device)
                    
                    # Note
                    # output variables used in "learned-based motion magnification" paper
                    Y, Va, Vb, _, _, Mb, Mb_ = self.model(amplified, frameA, frameB, frameC, amp_factor)
                    loss = self.criterion(Y, amplified) + self.tex_loss_w * self.criterion(Va, Vb) + self.sha_loss_w * self.criterion(Mb, Mb_)
                    val_loss += loss.item()

        return val_loss

    def _display_and_save_model(self, running_loss, val_loss, epoch, iter):
        print(running_loss)
        cal_train_loss = running_loss / 2000
        cal_val_loss = val_loss / self.val_size
        
        # print result
        print('[epoch: %d] train_loss: %.3f, val_loss: %.3f' % (epoch + 1, cal_train_loss, cal_val_loss))
        self.train_losses.append(cal_train_loss)
        self.val_losses.append(cal_val_loss)

        # save model
        torch.save({'epoch': epoch+1, 
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'train_loss': self.train_losses, 
                    'val_loss': self.val_losses}, 
                    self.PARA_PATH+'/epoch_{}_iter_{}.tar'.format(epoch+1, iter+1))

    def _forward_backward_propagation(self, train_loader, val_loader, epoch):
        # training
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            self.model.train()
            amplified, frameA, frameB, frameC, amp_factor = sample['amplified'].to(self.device), \
                                                            sample['frameA'].to(self.device), \
                                                            sample['frameB'].to(self.device), \
                                                            sample['frameC'].to(self.device), \
                                                            sample['mag_factor'].to(self.device)
            self.optimizer.zero_grad()

            # Note
            # output variables used in "learned-based motion magnification" paper
            Y, Va, Vb, _, _, Mb, Mb_ = self.model(amplified, frameA, frameB, frameC, amp_factor)
            loss = self.criterion(Y, amplified) + self.tex_loss_w * self.criterion(Va, Vb) + self.sha_loss_w * self.criterion(Mb, Mb_)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if i%2000 == 1999:
                # evaluation
                val_loss = self._get_val_loss(val_loader)
                self._display_and_save_model(running_loss, val_loss, epoch, i)
                running_loss = 0.0

    def inference(self, prev_frame, frame, amp_factor):
        """Run Magnification on two frames.
        Args:
            prev_frame: path to first frame
            frame: path to second frame
            amplification_factor: float for amplification factor
        """
        # convert image to tensor until 200 line
        prev_frame = Image.open(prev_frame)
        frame = Image.open(frame)
        prev_frame = prev_frame.resize((384,384))
        frame = frame.resize((384,384))
        prev_frame = np.asarray(prev_frame, dtype = 'float32') / 127.5 - 1.0
        frame = np.asarray(frame, dtype='float32') / 127.5 - 1.0
        amp_factor = np.array(amp_factor, dtype = 'float32')
        sample = {'prev_frame': prev_frame, 'frame': frame, 'mag_factor': amp_factor}
        sample = ToTensor()(sample, istrain=False)
        prev_frame = sample['prev_frame'].unsqueeze(0).to(self.device)
        frame = sample['frame'].unsqueeze(0).to(self.device)
        mag_factor = sample['mag_factor'].to(self.device)

        # inference a magnified image
        texture_a, shape_a = self.model.encoder(prev_frame)
        texture_b, shape_b = self.model.encoder(frame)
        out_shape_enc = self.model.res_manipulator(shape_a, shape_b, mag_factor)
        out = self.model.decoder(texture_b, out_shape_enc)
        return out

    def play(self, vid_dir, frame_ext, out_dir, amplification_factor, velocity_mag=False):
        """Magnify a video in the two-frames mode.
        Args:
            vid_dir: directory containing video frames videos are processed
                in sorted order.
            out_dir: directory to place output frames and resulting video.
            amplification_factor: the amplification factor,
                with 0 being no change.
            velocity_mag: if True, process video in Dynamic mode.
        """
        self._load()
        vid_name = os.path.basename(out_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        vid_frames = sorted(glob(os.path.join(vid_dir, '*.' + frame_ext)))
        
        if velocity_mag:
            print("Running in Dynamic mode")
        else:
            print("Running in Static mode")

        prev_frame = first_frame
        desc = vid_name if len(vid_name) < 10 else vid_name[:10]
        for frame in tqdm(vid_frames, desc=desc):
            file_name = os.path.basename(frame)
            out_amp = self.inference(prev_frame, frame, amplification_factor)
            im_path = os.path.join(out_dir, file_name)
            save_images(out_amp, im_path)
            if velocity_mag:
                prev_frame = frame

        # Try to combine it into a video
        subprocess.call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', '30', '-i',
              os.path.join(out_dir, '%06d.png'), '-c:v', 'libx264',
              os.path.join(out_dir, vid_name + '.mp4')]
            )

    def train(self, datapath):
        self._load()
        _, _, train_loader, val_loader = self.get_data(datapath)
        torch.cuda.empty_cache()
        for epoch in tqdm(range(self.remain_epoch)):
            self._forward_backward_propagation(train_loader, val_loader, epoch)
            torch.cuda.empty_cache()

    def play_temporal(self, vid_dir, frame_ext, out_dir, amp_factor, fl, fh, fs, n_filter_tap, filter_type):
      """Magnify video with a temporal filter.

      Args:
          vid_dir: directory containing video frames videos are processed
              in sorted order.
          out_dir: directory to place output frames and resulting video.
          amplification_factor: the amplification factor,
              with 0 being no change.
          fl: low cutoff frequency.
          fh: high cutoff frequency.
          fs: sampling rate of the video.
          n_filter_tap: number of filter tap to use.
          filter_type: Type of filter to use. Can be one of "butter", or "differenceOfIIR".
          For "differenceOfIIR", fl and fh specifies rl and rh coefficients as in Wadhwa et al.

          # Note 
          Not yet construct the FIR filter
      """
        self._load()
        nyq = fs / 2.0
        if filter_type == 'fir':
            filter_b = firwin(n_filter_tap, [fl, fh], nyq=nyq, pass_zero=False)
            filter_a = []
        elif filter_type == 'butter':
            filter_b, filter_a = butter(n_filter_tap, [fl/nyq, fh/nyq],
                                        btype='bandpass')
            filter_a = filter_a[1:]
        elif filter_type == 'differenceOfIIR':
            # This is a copy of what Neal did. Number of taps are ignored.
            # Treat fl and fh as rl and rh as in Wadhwa's code.
            # Write down the difference of difference equation in Fourier
            # domain to proof this:
            filter_b = [fh - fl, fl - fh]
            filter_a = [-1.0*(2.0 - fh - fl), (1.0 - fl) * (1.0 - fh)]
        else:
            raise ValueError('Filter type must be either '
                                '["fir", "butter", "differenceOfIIR"] got ' + \
                                filter_type)
        head, tail = os.path.split(out_dir)
        tail = tail + '_fl{}_fh{}_fs{}_n{}_{}'.format(fl, fh, fs,
                                                        n_filter_tap,
                                                        filter_type)
        out_dir = os.path.join(head, tail)
        vid_name = os.path.basename(out_dir)
        if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
        vid_frames = sorted(glob(os.path.join(vid_dir, '*.' + frame_ext)))

        if len(filter_a) is not 0:
            x_state = []
            y_state = []

            for frame in tqdm(vid_frames, desc='Applying IIR'):
                file_name = os.path.basename(frame)
                frame_no, _ = os.path.splitext(file_name)
                frame_no = int(frame_no)

                # read data until 321 line
                frame = Image.open(frame)
                frame = frame.resize((384,384))
                frame = np.asarray(frame, dtype='float32') / 127.5 - 1.0
                amp_factor = np.array(amp_factor, dtype = 'float32')

                # Caution
                # some sample names and variables do not match.
                sample = {'prev_frame': frame, 'frame': frame, 'mag_factor': amp_factor}
                sample = ToTensor()(sample, istrain=False)
                frame = sample['frame'].unsqueeze(0).to(self.device)
                mag_factor = sample['mag_factor'].to(self.device)

                # get texture, shape representation
                texture_enc, x = self.model.encoder(frame)
                x = ToNumpy()(x[0, :, :, :])
                x = np.expand_dims(x, axis = 0)
                x_state.insert(0, x)

                # set up initial condition.
                while len(x_state) < len(filter_b):
                    x_state.insert(0, x)
                if len(x_state) > len(filter_b):
                    x_state = x_state[:len(filter_b)]
                y = np.zeros_like(x)
                for i in range(len(x_state)):
                    y += x_state[i] * filter_b[i]
                for i in range(len(y_state)):
                    y -= y_state[i] * filter_a[i]

                # update y state
                y_state.insert(0, y)
                if len(y_state) > len(filter_a):
                    y_state = y_state[:len(filter_a)]
                x = np.squeeze(x)
                y = np.squeeze(y)

                # Caution
                # some sample names and variables do not match.
                # x means ref_shape_enc
                # y means filtered_enc
                sample = {'prev_frame': x, 'frame': y, 'mag_factor': amp_factor}
                sample = ToTensor()(sample, istrain=False)
                ref_shape_enc = sample['prev_frame'].unsqueeze(0).to(self.device)
                filtered_enc = sample['frame'].unsqueeze(0).to(self.device)
                mag_factor = sample['mag_factor'].to(self.device)
                
                # manipulate the filtered reprsentation
                out_enc = self.model.res_manipulator(0.0, filtered_enc, mag_factor)
                out_enc += ref_shape_enc - filtered_enc
                texture_enc = texture_enc.to(self.device)
                out_enc = out_enc.to(self.device)
                out = self.model.decoder(texture_enc, out_enc)
                
                # save image
                im_path = os.path.join(out_dir, file_name)
                im_path = os.path.join(out_dir, file_name)
                self._save_images(out, im_path)
        else:
            print("Not yet construct the FIR filter")
            return

        # Try to combine it into a video
        subprocess.call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', '30', '-i',
                os.path.join(out_dir, '%06d.png'), '-c:v', 'libx264',
                os.path.join(out_dir, vid_name + '.mp4')]
            )