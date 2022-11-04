import os
import itertools

import torch
from torch import nn
from torchaudio.models.conv_tasnet import MaskGenerator

from lib.utils import get_logger, soft_mask
from lib.transforms import Transform


class Model(nn.Module):
    MIXING_MATRICES_CACHE = {}

    def __init__(self, config, num_sources=None):
        super().__init__()

        self.args = locals()
        self.logger = get_logger('model')
        self.config = config

        self.num_sources = self.config.num_sources if num_sources is None else num_sources

        self.transform = Transform(
            stft_frame_size=self.config.stft_frame_size,
            stft_hop_size=self.config.stft_hop_size,
            device=self.config.device,
        )

        self.mask_generator = MaskGenerator(
            input_dim=self.config.num_frequency_bins,
            num_sources=self.num_sources,
            kernel_size=3,
            num_feats=128,
            num_hidden=512,
            num_layers=4,
            num_stacks=3,
            msk_activate='sigmoid',
        )

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path)
        if 'device' in checkpoint.keys():
            del checkpoint['device'] ## FIXME: remove
        if device:
            checkpoint['config'].device = device
        state_dict = checkpoint.pop('state_dict')

        instance = cls(**checkpoint)
        instance.load_state_dict(state_dict)
        instance.to(instance.config.device)

        return instance

    def save(self, path):
        checkpoint = self.args.copy()
        del checkpoint['self']
        del checkpoint['__class__']
        checkpoint['config'] = self.config
        checkpoint['state_dict'] = self.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def forward(self, x_true_wave):
        x_true_mag, x_true_phase = self.transform.stft(x_true_wave)
        m_pred_mag = self.mask_generator(x_true_mag.squeeze(1))
        s_pred_mag = soft_mask(m_pred_mag, x_true_mag)
        s_pred_wave = self.transform.istft(
            mag=s_pred_mag,
            phase=x_true_phase,
            length=x_true_wave.size(-1)
        )
        return s_pred_wave

    def generate_mixing_matrices(self, num_targets, max_sources, num_mix=None, allow_empty=False):
        parameters = locals()
        del parameters['self']

        def do_generate():
            output_perms = itertools.product([0, 1], repeat=max_sources)
            if num_mix is not None:
                output_perms = [perm for perm in output_perms if sum(perm) == num_mix]
            target_perms = list(itertools.product(output_perms, repeat=num_targets))
            perm_list = []
            for target_perm in target_perms:
                perm_sum = torch.tensor(target_perm).sum(dim=0)
                if (perm_sum <= 1).all() if allow_empty else (perm_sum == 1).all():
                    perm_list.append(target_perm)
            self.logger.info('mixing matrices are generated with %d permutations for parameters %s',
                             len(perm_list), parameters)
            return torch.tensor(perm_list).float().to(self.config.device)

        cache_key = '_'.join(str(v) for k, v in parameters.items())
        try:
            r = Model.MIXING_MATRICES_CACHE[cache_key]
        except KeyError:
            r = Model.MIXING_MATRICES_CACHE[cache_key] = do_generate()
        return r
