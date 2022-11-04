import os
import argparse
from types import SimpleNamespace

import torch

from lib.utils import CONFIG_FILENAME, BEST_CHECKPOINT_FILENAME, BEST_METRICS_FILENAME, configure_console_logger, \
    get_logger, default, build_run_name, ensure_clean_results_dir, setup_determinism, metrics_to_str
from lib.data.dataloader_utils import get_dataset_specs, create_dataloader
from lib.models import Model
from lib.trainers import get_trainer


class Test:
    def __init__(self, train_results_dir, librimix_root=None, realm_root=None,
                 eval_method=None,
                 eval_blind_num_repeat=None,
                 seed=None,
                 device_name=None):
        args = locals()
        del args['self']

        self.dataset_name, self.dataset_root = get_dataset_specs(librimix_root, realm_root)
        run_name = build_run_name(
            args=args,
            prepend_items={'test': self.dataset_name},
            exclude_keys=['train_results_dir', 'librimix_root', 'realm_root']
        )
        self.results_dir = os.path.join(train_results_dir, 'test', run_name)

        self.config = SimpleNamespace()
        self.config.results_dir = self.results_dir
        self.config.train_results_dir = train_results_dir
        self.config.librimix_root = librimix_root
        self.config.realm_root = realm_root
        self.config.eval_method = default(eval_method, None)
        self.config.eval_blind_num_repeat = default(eval_blind_num_repeat, None)
        self.config.seed = default(seed, None)
        self.config.device_name = default(device_name, 'cuda')

        self.logger = None
        self.config.device = torch.device(self.config.device_name)

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)
        setup_determinism(self.config.seed)
        self.logger = get_logger('test', self.config.results_dir)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.results_dir, CONFIG_FILENAME))

        model = Model.load(
            path=os.path.join(self.config.train_results_dir, BEST_CHECKPOINT_FILENAME),
            device=self.config.device
        ).eval()

        if self.config.eval_method:
            model.config.eval_method = self.config.eval_method

        if self.config.eval_blind_num_repeat:
            model.config.eval_blind_num_repeat = self.config.eval_blind_num_repeat

        if model.config.eval_method == 'blind':
            model_name = 'mixcycle'
            partition = 'validation'
            batch_size = 128
            shuffle = True
        elif model.config.eval_method == 'reference-valid':
            model_name = model.config.model_name
            partition = 'validation'
            batch_size = 128
            shuffle = False
        else:
            model_name = model.config.model_name
            partition = 'testing'
            batch_size = 1
            shuffle = False

        dataloader = create_dataloader(
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            partition=partition,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.logger.info('using %d samples for evaluation', len(dataloader.dataset))

        trainer = get_trainer(model_name)(model=model)
        with torch.inference_mode():
            metrics = trainer.validate(dataloader)

        self.logger.info('[TEST] %s', metrics_to_str(metrics))

        torch.save(metrics, os.path.join(self.config.results_dir, BEST_METRICS_FILENAME))

        self.logger.info('completed')


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-results-dir', type=str, required=True)
    arg_parser.add_argument('--librimix-root', type=str)
    arg_parser.add_argument('--realm-root', type=str)
    arg_parser.add_argument('--eval-method', choices=['reference', 'reference-valid', 'blind'])
    arg_parser.add_argument('--eval-blind-num-repeat', type=int)
    arg_parser.add_argument('--seed', type=int)
    arg_parser.add_argument('--device-name', type=str)

    cmd_args = arg_parser.parse_args()
    Test(**vars(cmd_args)).start()
