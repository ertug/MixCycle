import os
import argparse
from types import SimpleNamespace

import torch

from lib.utils import BEST_METRICS_FILENAME, configure_console_logger, get_logger, default, build_run_name, \
    ensure_clean_results_dir, soft_mask, MetricAccumulator, metrics_to_str
from lib.data.dataloader_utils import get_dataset_specs, create_dataloader
from lib.transforms import Transform
from lib.losses import sisnri


class IdealRatioMask:
    def __init__(self, irm_results_root, librimix_root=None, realm_root=None,
                 stft_frame_size=None,
                 stft_hop_size=None,
                 device_name=None):
        args = locals()
        del args['self']

        self.dataset_name, self.dataset_root = get_dataset_specs(librimix_root, realm_root)
        run_name = build_run_name(
            args=args,
            prepend_items={'irm': self.dataset_name},
            exclude_keys=['irm_results_root', 'sc09mix_root', 'librimix_root']
        )
        self.results_dir = os.path.join(irm_results_root, run_name)

        self.config = SimpleNamespace()
        self.config.results_dir = self.results_dir
        self.config.librimix_root = librimix_root
        self.config.realm_root = realm_root
        self.config.stft_frame_size = default(stft_frame_size, 512)
        self.config.stft_hop_size = default(stft_hop_size, 128)
        self.config.device_name = default(device_name, 'cuda')

        self.logger = None
        self.config.device = torch.device(self.config.device_name)

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)
        self.logger = get_logger('irm')
        self.logger.info('config: %s', self.config)

        dataloader = create_dataloader(
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            partition='testing',
            batch_size=1,
        )
        self.logger.info('using %d samples for evaluation', len(dataloader.dataset))

        transform = Transform(
            stft_frame_size=self.config.stft_frame_size,
            stft_hop_size=self.config.stft_hop_size,
            device=self.config.device,
        )

        with torch.inference_mode():
            sisnri_accumulator = MetricAccumulator()

            for x_true_wave, s_true_wave in dataloader:
                x_true_wave = x_true_wave.to(self.config.device)
                s_true_wave = s_true_wave.to(self.config.device)

                x_true_mag, x_true_phase = transform.stft(x_true_wave)
                s_true_mag, _ = transform.stft(s_true_wave)
                s_pred_mag = soft_mask(s_true_mag, x_true_mag)
                s_pred_wave = transform.istft(
                    mag=s_pred_mag,
                    phase=x_true_phase,
                    length=x_true_wave.size(-1)
                )

                batch_sisnri = sisnri(
                    true_wave=s_true_wave,
                    pred_wave=s_pred_wave,
                    x_true_wave=x_true_wave,
                )

                sisnri_accumulator.store(batch_sisnri)

        std, mean = sisnri_accumulator.std_mean()
        metrics = {
            'sisnri': mean.item(),
            'sisnri_std': std.item()
        }

        metrics_str = metrics_to_str(metrics)
        self.logger.info('[IRM] %s', metrics_str)

        torch.save(metrics, os.path.join(self.config.results_dir, BEST_METRICS_FILENAME))

        self.logger.info('completed')


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--irm-results-root', type=str, required=True)
    arg_parser.add_argument('--librimix-root', type=str)
    arg_parser.add_argument('--realm-root', type=str)
    arg_parser.add_argument('--stft-frame-size', type=int)
    arg_parser.add_argument('--stft-hop-size', type=int)
    arg_parser.add_argument('--device-name', type=str)

    cmd_args = arg_parser.parse_args()
    IdealRatioMask(**vars(cmd_args)).start()
