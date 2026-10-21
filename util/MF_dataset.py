# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640, transform=None):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform or []
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        # The official train.txt lists 784 additional ``*_flip`` samples, but
        # some dataset archives only contain the 784 source files.  Generate
        # those prescribed samples lazily so every epoch still sees all 1568
        # entries used by the paper.
        should_flip = name.endswith('_flip')
        source_name = name[:-5] if should_flip else name
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, source_name))
        # np.asarray(PIL.Image) is read-only with recent Pillow versions.
        image     = np.array(Image.open(file_path), copy=True) # (h,w,c)
        if should_flip:
            image = np.ascontiguousarray(image[:, ::-1])
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(
            Image.fromarray(image).resize((self.input_w, self.input_h), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).transpose((2,0,1))/255
        # Class IDs are categorical.  Interpolating an int32/PIL-I mask with
        # Pillow's default filter can create invalid labels such as 9 and 10.
        label = np.asarray(
            Image.fromarray(label).resize((self.input_w, self.input_h), resample=Image.Resampling.NEAREST),
            dtype=np.int64,
        )

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        image = np.asarray(
            Image.fromarray(image).resize((self.input_w, self.input_h), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).transpose((2,0,1))/255

        return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/MF/'
    MF_dataset()
