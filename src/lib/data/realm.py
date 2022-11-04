import hashlib
from pathlib import Path

from torch.utils.data import Dataset
import torchaudio


class RealM(Dataset):
    VALIDATION_PERCENTAGE = 20
    MAX_NUM_FILES = 2 ** 27 - 1  # ~134M
    SOURCE_SUFFIXES = ['source1hat.wav', 'source2hat.wav']

    def __init__(self, root, partition):
        self.mix_root = Path(root) / 'audio_files_converted_8000Hz'
        self.src_root = Path(root) / 'separations'
        self.mix_paths = []

        if partition == 'testing':
            partition = 'validation'

        for file_path in self.mix_root.glob('*/*.wav'):
            if self._assign_partition(file_path.name) == partition:
                self.mix_paths.append(file_path)

        self.mix_paths.sort()

    def __len__(self):
        return len(self.mix_paths)

    def __getitem__(self, idx):
        mix_path = self.mix_paths[idx]
        mix = self._load_audio(mix_path)
        srcs = [self._load_audio(self.src_root / (mix_path.name + s)) for s in self.SOURCE_SUFFIXES]
        return None, mix, srcs

    def _load_audio(self, file_path):
        wave, sample_rate = torchaudio.load(file_path)
        assert sample_rate == 8000
        return wave

    def _assign_partition(self, filename):
        filename_hashed = hashlib.sha1(filename.encode('ascii')).hexdigest()
        percentage_hash = ((int(filename_hashed, 16) %
                            (self.MAX_NUM_FILES + 1)) *
                           (100.0 / self.MAX_NUM_FILES))
        if filename.startswith('early'):
            return 'discarded'
        elif percentage_hash < self.VALIDATION_PERCENTAGE:
            return 'validation'
        else:
            return 'training'
