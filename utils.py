# last 3 functions from https://github.com/Newmu/dcgan_code
# References
# 1. https://wikidocs.net/57165
# 2. https://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html#Sampler

import torch
import random
import numpy as np

from PIL import Image
from torch.utils.data.sampler import Sampler

class ToTensor(object):
  def __call__(self, sample):
    amplified, frameA, frameB, frameC, mag_factor = sample['amplified'], sample['frameA'], sample['frameB'], sample['frameC'], sample['mag_factor']
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    amplified = amplified.transpose((2, 0, 1))
    frameA = frameA.transpose((2, 0, 1))
    frameB = frameB.transpose((2, 0, 1))
    frameC = frameC.transpose((2, 0, 1))

    # convert tensor
    amplified = torch.from_numpy(amplified)
    frameA = torch.from_numpy(frameA)
    frameB = torch.from_numpy(frameB)
    frameC = torch.from_numpy(frameC)
    mag_factor = torch.from_numpy(mag_factor)
    mag_factor = mag_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    ToTensor_sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_factor': mag_factor}
    return ToTensor_sample

class ToNumpy(object):
  def __call__(self, amplified):
    # amplified, frameA, frameB, frameC, mag_factor = sample['amplified'], sample['frameA'], sample['frameB'], sample['frameC'], sample['mag_factor']
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    amplified = amplified.permute((1, 2, 0))
    #frameA = frameA.transpose((2, 0, 1))
    #frameB = frameB.transpose((2, 0, 1))
    #frameC = frameC.transpose((2, 0, 1))

    # convert tensor
    amplified = amplified.numpy()
    #frameA = torch.from_numpy(frameA)
    #frameB = torch.from_numpy(frameB)
    #frameC = torch.from_numpy(frameC)
    #mag_factor = torch.from_numpy(mag_factor)
    #mag_factor = mag_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    #ToTensor_sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_factor': mag_factor}
    print(amplified.shape)
    return amplified

class shot_noise(object):
  # This function approximate poisson noise upto 2nd order.
  def __init__(self, n):
    self.n = n

  def _get_shot_noise(self, image):
    n = torch.zeros_like(image).normal_(mean=0.0, std=1.0)
    # strength ~ sqrt image value in 255, divided by 127.5 to convert
    # back to -1, 1 range.

    n_str = torch.sqrt(torch.as_tensor(image + 1.0)) / torch.sqrt(torch.as_tensor(127.5))
    return torch.mul(n, n_str)

  def _preproc_shot_noise(self, image, n):
    nn = np.random.uniform(0, n)
    return image + nn * self._get_shot_noise(image)

  def __call__(self, sample):
    amplified, frameA, frameB, frameC, mag_factor = sample['amplified'], sample['frameA'], sample['frameB'], sample['frameC'], sample['mag_factor']
    # add shot noise
    frameA = self._preproc_shot_noise(frameA, self.n)
    frameB = self._preproc_shot_noise(frameB, self.n)
    frameC = self._preproc_shot_noise(frameC, self.n)

    preproc_sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_factor': mag_factor}
    return preproc_sample

class num_sampler(Sampler):
# Sampling a specific number of multiple-th indices from data.
  def __init__(self, data, is_val=True, shuffle=False, num=10):
    self.num_samples = len(data)
    self.is_val = is_val
    self.shuffle = shuffle
    self.num = num

  def __iter__(self):
    k = []
    for i in range(self.num_samples):
      if self.is_val: # case of validation dataset
        if i%self.num == self.num-1:
          k.append(i)
      else: # case of train dataset
        if i%self.num != self.num-1:
          k.append(i)

    if self.shuffle:
      random.shuffle(k)
    return iter(k)

  def __len__(self):
    return self.num_samples

def inverse_transform(image):
    return (image + 1.) / 2.

def imsave(im, path):
    if issubclass(im.dtype.type, np.floating):
        im = im * 255
        im = im.astype('uint8')
    im = Image.fromarray(im)
    return im.save(path,"PNG")

def save_image(image, image_path):
    return imsave(inverse_transform(image), image_path)    