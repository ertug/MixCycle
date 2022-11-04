import os
import re
import hashlib

import torch
from torch.utils.data import Dataset
import librosa

from lib.utils import CONFIG_FILENAME


SAMPLE_FILENAME_FORMAT = '{:07d}.pth'


class SC09(Dataset):
    PARTITIONS = ('training', 'validation', 'testing')
    CLASSES = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    ORIGINAL_SAMPLE_RATE = 16000
    VALIDATION_PERCENTAGE = 10
    TESTING_PERCENTAGE = 10
    TRAINING_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TESTING_PERCENTAGE
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    def __init__(self, root, partition, classes=None, sample_rate=ORIGINAL_SAMPLE_RATE):
        if partition not in self.PARTITIONS:
            raise Exception(f'invalid partition: {partition}')

        self.sample_rate = sample_rate
        self.metadata = []

        for cls in (self.CLASSES if classes is None else classes):
            base = os.path.join(root, cls)
            filenames = sorted(os.listdir(base))
            for filename in filenames:
                if self._assign_partition(filename) == partition:
                    path = os.path.join(base, filename)
                    self.metadata.append((path, cls))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path, label = self.metadata[idx]
        wave = torch.zeros(self.sample_rate)
        x, _ = librosa.core.load(path, sr=self.sample_rate)

        # center if shorter than one second
        centered_start = (self.sample_rate-x.shape[0])//2
        centered_end = centered_start + x.shape[0]
        wave[centered_start:centered_end] = torch.from_numpy(x)

        return wave, label

    def _assign_partition(self, filename):
        """ copied from the dataset README """
        base_name = os.path.basename(filename)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        hash_name_hashed = hashlib.sha1(hash_name.encode('ascii')).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (self.MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / self.MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < self.VALIDATION_PERCENTAGE:
            result = 'validation'
        elif percentage_hash < (self.TESTING_PERCENTAGE + self.VALIDATION_PERCENTAGE):
            result = 'testing'
        else:
            result = 'training'
        return result


class SC09Mix(Dataset):
    PARTITIONS = SC09.PARTITIONS

    def __init__(self, root, partition):
        assert partition in SC09Mix.PARTITIONS

        self.mix_dir = os.path.join(root, 'mix', partition)
        self.count = len(os.listdir(self.mix_dir))
        self.config = torch.load(os.path.join(root, CONFIG_FILENAME))

    def _load_sample(self, idx):
        return torch.load(os.path.join(self.mix_dir, SAMPLE_FILENAME_FORMAT.format(idx)))

    def _get_single_item(self, idx):
        sample = self._load_sample(idx)

        return sample['mixture'].unsqueeze(0), sample['sources']

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self._get_single_item(idx)
