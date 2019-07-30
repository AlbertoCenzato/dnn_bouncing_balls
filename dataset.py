import math
import re
import os

import numpy as np

from torch.utils.data import Dataset


class SutskeverDataset(Dataset):
    """
    Loads from folder 'path' the dataset generated with 'dataset_generation.py'
    as numpy.ndarray.
    Expects one .npy file for sequence and returns numpy.ndarrays with shape
    (time_lenght, height, width, channels).
    Raises OSError either if path does not exist or is not a directory
    or is empty.
    """

    def __init__(self, path, transform=None, filename_regex='^sequence_[0-9]+\\.npy$'):
        super(SutskeverDataset, self).__init__()
        if not os.path.exists(path):
            raise OSError("Path {} does not exist".format(path))
        if not os.path.isdir(path):
            raise OSError("{} is not a folder".format(path))

        dir_files = os.listdir(path)
        if len(dir_files) == 0:
            raise OSError("Directory {} is empty".format(path))

        self._sample_shape = None
        self._path = path
        self._transform = transform

        regex = re.compile(filename_regex)
        self._files = []
        for file in dir_files:
            result = regex.search(file)
            if result:
                self._files.append(file)
        self._dataset_size = len(self._files)


    def __len__(self):
        return self._dataset_size

    def __getitem__(self, key):
        file_path = os.path.join(self._path, self._files[key])
        sample = np.load(file_path)

        shape = self.get_sample_shape()
        sample = sample.reshape(shape)

        if self._transform:
            sample = self._transform(sample)

        return sample

    def get_sample_shape(self):
        """
        :return: the shape of a dataset element as a tuple
        """
        if not self._sample_shape:
            file_path = os.path.join(self._path, self._files[0])
            sample = np.load(file_path)

            height = int(math.sqrt(sample.shape[1]))
            self._sample_shape = (sample.shape[0], height, height, 1)
        return self._sample_shape
