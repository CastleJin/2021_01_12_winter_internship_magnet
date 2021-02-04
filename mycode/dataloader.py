# References
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

#from __future__ import print_function

import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class read_and_decode_3frame(Dataset):
  def __init__(self, data_path, transform = None):
    self.data_path = data_path # root of dataset
    self.sub_dir = os.listdir(self.data_path) # sub directory
    
    # data path
    self.amplified_path = os.path.join(self.data_path, self.sub_dir[0])
    self.frameA_path = os.path.join(self.data_path, self.sub_dir[1])
    self.frameB_path = os.path.join(self.data_path, self.sub_dir[2])
    self.frameC_path = os.path.join(self.data_path, self.sub_dir[3])
    self.meta_path = os.path.join(self.data_path, self.sub_dir[4])
    self.transform = transform

  
  def _read_json(self, path):
    with open(path,'r') as f:
      json_data = json.load(f)
    return json_data['amplification_factor']

  def __len__(self):
    file_list = os.listdir(self.amplified_path)
    return len(file_list)
  
  def __getitem__(self,idx):
    # subfile list
    amplified_list = os.listdir(self.amplified_path)
    frameA_list = os.listdir(self.frameA_path)
    frameB_list = os.listdir(self.frameB_path)
    frameC_list = os.listdir(self.frameC_path)
    meta_list = os.listdir(self.meta_path)

    # sort
    amplified_list.sort()
    frameA_list.sort()
    frameB_list.sort()
    frameC_list.sort()
    meta_list.sort()
  
    # read image & json
    amplified = Image.open(os.path.join(self.amplified_path, amplified_list[idx]))
    frameA = Image.open(os.path.join(self.frameA_path, frameA_list[idx]))
    frameB = Image.open(os.path.join(self.frameB_path, frameB_list[idx]))
    frameC = Image.open(os.path.join(self.frameC_path, frameC_list[idx]))
    mag_factor = self._read_json(os.path.join(self.meta_path, meta_list[idx]))

    # convert nparray & normalize to -1 to 1
    amplified = np.array(amplified, dtype = 'float32') / 127.5 - 1.0
    frameA = np.array(frameA, dtype = 'float32') / 127.5 - 1.0
    frameB = np.array(frameB, dtype = 'float32') / 127.5 - 1.0
    frameC = np.array(frameC, dtype = 'float32') / 127.5 - 1.0
    mag_factor -= 1.0
    mag_factor = np.array(mag_factor, dtype = 'float32')

    sample = {'amplified': amplified, 'frameA': frameA, 'frameB': frameB, 'frameC': frameC, 'mag_factor': mag_factor}

    if self.transform is not None:
      sample = self.transform(sample)

    return sample
