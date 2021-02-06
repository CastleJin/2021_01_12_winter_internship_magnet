import os
import os.path as osp
import sys
from PIL import Image
import six
import string
import numpy as np

import lmdb
import pickle
import umsgpack
import pyarrow as pa
from os.path import basename

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = osp.join(db_path, 'train.lmdb')
        self.mag_path = osp.join(db_path, 'train_mf.txt')
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        # read mag
        self.amp_factor = []
        with open(self.mag_path, 'r') as f:
          lines = f.readlines()
          for line in lines:
              self.amp_factor.append(line)
    
        self.transform = transform

    def __getitem__(self, index):
        # read img
        images = []
        for i in range(3):
          images.append(None)
        env = self.env
        with env.begin(write=False) as txn:
            #print("key{}".format(self.keys[index].decode("ascii")))
            byteflow = txn.get(self.keys[index])
        unpacked = loads_pyarrow(byteflow)
  
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        
        amp = Image.open(buf).convert('RGB')

        imgbuf = unpacked[1]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        
        A = Image.open(buf).convert('RGB')

        imgbuf = unpacked[2]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        
        B = Image.open(buf).convert('RGB')

        imgbuf = unpacked[3]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        
        C = Image.open(buf).convert('RGB')

        amp = np.array(amp, dtype = 'float32') / 127.5 - 1.0
        A = np.array(A, dtype = 'float32') / 127.5 - 1.0
        B = np.array(B, dtype = 'float32') / 127.5 - 1.0
        C = np.array(C, dtype = 'float32') / 127.5 - 1.0
        mag_factor = np.array(self.amp_factor[index], dtype = 'float32')

        sample = {'amplified': amp, 'frameA': A, 'frameB': B, 'frameC': C, 'mag_factor': mag_factor}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'