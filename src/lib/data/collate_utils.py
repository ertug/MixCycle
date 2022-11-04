from typing import Tuple, List
from collections import namedtuple

import torch

#########################################################################################################
# based on https://github.com/pytorch/audio/blob/main/examples/source_separation/utils/dataset/utils.py #
#########################################################################################################

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]
Batch = namedtuple("Batch", ["mix", "src"])


def _fix_num_frames(sample: SampleType, target_num_frames: int, sample_rate: int, random_start=False):
    """Ensure waveform has exact number of frames by slicing or padding"""
    mix = sample[1]  # [1, time]
    src = torch.cat(sample[2], 0)  # [num_sources, time]

    num_channels, num_frames = src.shape
    num_seconds = torch.div(num_frames, sample_rate, rounding_mode="floor")
    target_seconds = torch.div(target_num_frames, sample_rate, rounding_mode="floor")
    if num_frames >= target_num_frames:
        if random_start and num_frames > target_num_frames:
            start_frame = torch.randint(num_seconds - target_seconds + 1, [1]) * sample_rate
            mix = mix[:, start_frame:]
            src = src[:, start_frame:]
        mix = mix[:, :target_num_frames]
        src = src[:, :target_num_frames]
        mask = torch.ones_like(mix)
    else:
        num_padding = target_num_frames - num_frames
        pad = torch.zeros([1, num_padding], dtype=mix.dtype, device=mix.device)
        mix = torch.cat([mix, pad], 1)
        src = torch.cat([src, pad.expand(num_channels, -1)], 1)
        mask = torch.ones_like(mix)
        mask[..., num_frames:] = 0
    return mix, src, mask


def collate_fn_wsj0mix_train(samples, sample_rate, duration):
    target_num_frames = int(duration * sample_rate)

    mixes, srcs = [], []
    for sample in samples:
        mix, src, _ = _fix_num_frames(sample, target_num_frames, sample_rate, random_start=True)

        mixes.append(mix)
        srcs.append(src)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0))


def collate_fn_wsj0mix_test(samples):
    assert len(samples) == 1

    mixes, srcs = [], []
    for sample in samples:
        mix = sample[1]  # [1, time]
        src = torch.cat(sample[2], 0)  # [num_sources, time]

        mixes.append(mix)
        srcs.append(src)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0))
