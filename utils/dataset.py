import torch.utils.data as data
from PIL import Image
import os
import numpy as np


class VideoRecord(object):
    def __init__(self, root_path, row):
        self._data = [os.path.join(root_path, row[0]), row[1]]

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class DataSetPol(data.Dataset):
    def __init__(self, root_path, list_file, modality='RGB', transform=None, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality

        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(directory).convert('L')]
        elif self.modality == "Flow":
            return [Image.open(directory).convert('L')]

    def _parse_list(self):
        self.video_list = [VideoRecord(self.root_path, x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record)

    def get(self, record):
        images = self._load_image(record.path)
        #print(record.path, record.label)
        
        a = np.random.randint(0, 4)
        #print(a)
        process_data = self.transform([images[0], a])
        #import pdb; pdb.set_trace()
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class DataSetSeg(data.Dataset):
    def __init__(self, root_path, list_file, modality='RGB', transform=None, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality

        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(directory).convert('L')]
        elif self.modality == "Flow":
            return [Image.open(directory).convert('L')]

    def _parse_list(self):
        self.video_list = [VideoRecord(self.root_path, x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record)

    def get(self, record):
        a = np.random.randint(0, 4)
        inp_imgs = self._load_image(record.path)
        #print(record.path, record.label)
        process_data = self.transform([inp_imgs[0], a])

        seg_imgs = self._load_image(record.path.replace('imgs', 'seg'))
        process_label = self.transform([seg_imgs[0], a])
        #import pdb; pdb.set_trace()
        return process_data, process_label

    def __len__(self):
        return len(self.video_list)