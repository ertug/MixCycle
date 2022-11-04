from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets.librimix import LibriMix

from lib.data.collate_utils import collate_fn_wsj0mix_train, collate_fn_wsj0mix_test
from lib.data.realm import RealM


SAMPLE_RATE = 8000


def get_dataset_specs(librimix_root=None, realm_root=None):
    if librimix_root and realm_root:
        raise Exception('only one dataset root should be given')
    elif librimix_root:
        dataset_name = 'librimix'
        dataset_root = librimix_root
    elif realm_root:
        dataset_name = 'realm'
        dataset_root = realm_root
    else:
        raise Exception('at least one dataset root should be given')

    return dataset_name, dataset_root


def create_dataloader(dataset_name, dataset_root, partition, batch_size, subsample_ratio=1.0, shuffle=None):
    assert partition in ('training', 'validation', 'testing')

    if partition in ('training', 'validation'):
        collate_fn = partial(collate_fn_wsj0mix_train, sample_rate=SAMPLE_RATE, duration=3)
    else:
        collate_fn = partial(collate_fn_wsj0mix_test)

    if dataset_name == 'librimix':
        subset_mapping = {
            'training': 'train-360',
            'validation': 'dev',
            'testing': 'test',
        }
        dataset = LibriMix(
            root=dataset_root,
            subset=subset_mapping[partition],
            num_speakers=2,
            sample_rate=SAMPLE_RATE,
            task='sep_clean',
        )
    elif dataset_name == 'realm':
        dataset = RealM(
            root=dataset_root,
            partition=partition,
        )
    else:
        raise Exception(f'unknown dataset name: {dataset_name}')

    if subsample_ratio < 1.0:
        subsampled_dataset_size = int(len(dataset) // (1/subsample_ratio))
        indices = torch.randperm(len(dataset)).int()[:subsampled_dataset_size]
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(partition == 'training' or shuffle),
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=(partition == 'training'),
        pin_memory=True,
    )


def double_mixture_generator(dataloader):
    iterator = iter(dataloader)
    while True:
        try:
            x_true_wave_1, _ = next(iterator)
            x_true_wave_2, _ = next(iterator)
            yield x_true_wave_1, x_true_wave_2
        except StopIteration:
            return
