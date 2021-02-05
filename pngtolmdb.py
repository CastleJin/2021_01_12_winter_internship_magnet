# References
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

#from __future__ import print_function

import os
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp

class get(Dataset):
  def __init__(self, data_path, loader=None):
    amppath = osp.join(data_path, '1')
    Apath = osp.join(data_path, '2')
    Bpath = osp.join(data_path, '3')
    Cpath = osp.join(data_path, '4')
    self.amp = ImageFolder(amppath, loader=loader)
    self.A = ImageFolder(Apath, loader=loader)
    self.B = ImageFolder(Bpath, loader=loader)
    self.C = ImageFolder(Cpath, loader=loader)


  def __len__(self):
    return len(self.amp)
  
  def __getitem__(self,idx):
    return self.amp[idx][0], self.A[idx][0], self.B[idx][0], self.C[idx][0]


import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import umsgpack
import tqdm
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


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        img, idx = line.split(" ")
        map[img] = idx
    return map


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        images = []
        for i in range(3):
          images.append(None)
        env = self.env
        with env.begin(write=False) as txn:
            print("key{}".format(self.keys[index].decode("ascii")))
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

        import numpy as np
        amp = np.array(amp)
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        return amp, A, B , C

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", write_frequency=5000):
    all_imgpath = []
    all_idxs = []
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = get(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        amp, A, B, C = data[0]
        all_idxs.append(idx)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((amp, A, B, C)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

import fire

if __name__ == '__main__':
    fire.Fire(folder2lmdb)