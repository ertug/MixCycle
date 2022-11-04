import os
import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from lib.utils import CONFIG_FILENAME, configure_console_logger, get_logger, default, build_run_name, \
    ensure_clean_results_dir, setup_determinism
from lib.data.sc09 import SAMPLE_FILENAME_FORMAT, SC09


class Mixer:
    def __init__(self, sc_dir, mix_results_root,
                 num_train_samples=None,
                 num_test_samples=None,
                 num_components=None,
                 classes=None,
                 sample_rate=None,
                 normalization=None,
                 seed=None):
        args = locals()
        del args['self']
        run_name = build_run_name(
            args=args,
            prepend_items={'dataset': 'sc09mix'},
            exclude_keys=['sc_dir', 'mix_results_root']
        )
        self.results_dir = os.path.join(mix_results_root, run_name)

        self.config = SimpleNamespace()
        self.config.results_dir = self.results_dir
        self.config.sc_dir = sc_dir
        self.config.mix_results_root = mix_results_root
        self.config.num_train_samples = default(num_train_samples, 15000)
        self.config.num_test_samples = default(num_test_samples, 5000)
        self.config.num_components = default(num_components, 2)
        self.config.classes = default(classes, list(range(len(SC09.CLASSES))))
        self.config.sample_rate = default(sample_rate, 8000)
        self.config.normalization = default(normalization, 'standardize')
        self.config.seed = default(seed, None)
        self.config.sample_length = self.config.sample_rate  # 1 second recordings

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)
        setup_determinism(self.config.seed)

        self.logger = get_logger('sc09mix', self.config.results_dir)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.results_dir, CONFIG_FILENAME))

        for partition in SC09.PARTITIONS:
            num_samples = self.config.num_train_samples if partition == 'training' else self.config.num_test_samples
            self.logger.info('generating the %s partition...', partition)

            mix_dir = os.path.join(self.config.results_dir, 'mix', partition)
            os.makedirs(mix_dir, exist_ok=True)

            infinite_dataloader = self._create_infinite_dataloader(partition)

            for sample_idx in trange(num_samples):
                mixture = self._generate_mixture(infinite_dataloader)
                torch.save(mixture, os.path.join(mix_dir, SAMPLE_FILENAME_FORMAT.format(sample_idx)))

        self.logger.info('completed')

    def _generate_mixture(self, dataloader):
        mixture = torch.zeros(self.config.sample_length)
        sources = torch.zeros(self.config.num_components, self.config.sample_length)
        for i in range(self.config.num_components):
            source, _ = next(dataloader)
            source = source.view(-1)

            if self.config.normalization == 'none':
                pass
            elif self.config.normalization == 'rms':
                source = source / source.square().mean().sqrt()
            elif self.config.normalization == 'standardize':
                source = (source - source.mean()) / source.std()
            else:
                raise Exception(f'unknown normalization: {self.config.normalization}')

            mixture += source
            sources[i] = source

        return {
            'mixture': mixture,
            'sources': sources,
        }

    def _create_infinite_dataloader(self, partition):
        dataset = SC09(
            root=self.config.sc_dir,
            partition=partition,
            classes=[SC09.CLASSES[class_idx] for class_idx in self.config.classes],
            sample_rate=self.config.sample_rate
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        while True:
            for sample in dataloader:
                yield sample


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--sc-dir', type=str, required=True)
    arg_parser.add_argument('--mix-results-root', type=str, required=True)
    arg_parser.add_argument('--partition', choices=['training', 'validation', 'testing'])
    arg_parser.add_argument('--num-samples', type=int)
    arg_parser.add_argument('--num-components', type=int)
    arg_parser.add_argument('--classes', type=str, help='comma separated')
    arg_parser.add_argument('--sample-rate', type=int)
    arg_parser.add_argument('--normalization', choices=['none', 'rms', 'standardize'])
    arg_parser.add_argument('--seed', type=int)

    cmd_args = arg_parser.parse_args()
    if cmd_args.classes:
        cmd_args.classes = [int(cls) for cls in cmd_args.classes.split(',')]

    Mixer(**vars(cmd_args)).start()
