# Most codes from https://github.com/Lyken17/Efficient-PyTorch
# References
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import six
import string
import lmdb
import pickle
import umsgpack
import tqdm
import torch

import os.path as osp
import numpy as np
import pyarrow as pa
import torch.utils.data as data

from os.path import basename
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

class get(Dataset):
    # read a dataset from image folder
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