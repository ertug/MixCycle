import os
import argparse
from time import time
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils import CONFIG_FILENAME, BEST_CHECKPOINT_FILENAME, METRICS_HISTORY_FILENAME, configure_console_logger, \
    default, build_run_name, ensure_clean_results_dir, setup_determinism, get_logger, total_num_params
from lib.data.dataloader_utils import get_dataset_specs, create_dataloader
from lib.trainers import get_trainer
from lib.models import Model


class Training:
    def __init__(self, train_results_root, librimix_root=None, realm_root=None,
                 stft_frame_size=None,
                 stft_hop_size=None,
                 model_name=None,
                 model_load_path=None,
                 mixcycle_init_epochs=None,
                 snr_max=None,
                 train_batch_size=None,
                 valid_batch_size=None,
                 lr=None,
                 grad_clip=None,
                 train_subsample_ratio=None,
                 valid_subsample_ratio=None,
                 eval_method=None,
                 eval_blind_num_repeat=None,
                 eval_epochs=None,
                 patience=None,
                 min_epochs=None,
                 seed=None,
                 run_id=None,
                 device_name=None):
        args = locals()
        del args['self']

        self.dataset_name, self.dataset_root = get_dataset_specs(librimix_root, realm_root)
        run_name = build_run_name(
            args=args,
            prepend_items={'train': self.dataset_name},
            exclude_keys=['train_results_root', 'realm_root', 'librimix_root', 'model_load_path']
        )
        self.results_dir = os.path.join(train_results_root, run_name)

        self.config = SimpleNamespace()
        self.config.results_dir = self.results_dir
        self.config.stft_frame_size = default(stft_frame_size, 512)
        self.config.stft_hop_size = default(stft_hop_size, 128)
        self.config.model_name = default(model_name, 'mixcycle')
        self.config.model_load_path = default(model_load_path, None)
        self.config.mixcycle_init_epochs = default(mixcycle_init_epochs, 50)
        self.config.snr_max = default(snr_max, 30.0)
        self.config.train_batch_size = default(train_batch_size, 128)
        self.config.valid_batch_size = default(valid_batch_size, 128)
        self.config.lr = default(lr, 0.001)
        self.config.grad_clip = default(grad_clip, 5.0)
        self.config.train_subsample_ratio = default(train_subsample_ratio, 1.0)
        self.config.valid_subsample_ratio = default(valid_subsample_ratio, 1.0)
        self.config.eval_method = default(eval_method, 'reference')
        self.config.eval_blind_num_repeat = default(eval_blind_num_repeat, 1)
        self.config.eval_epochs = default(eval_epochs, 1)
        self.config.patience = default(patience, 50)
        self.config.min_epochs = default(min_epochs, None)
        self.config.seed = default(seed, None)
        self.config.device_name = default(device_name, 'cuda')

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)

        self.logger = get_logger('train', self.config.results_dir)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.results_dir, CONFIG_FILENAME))

        setup_determinism(self.config.seed)

        self.config.device = torch.device(self.config.device_name)
        self.tensorboard = SummaryWriter(os.path.join(self.config.results_dir, 'tb'))

        self.cur_epoch = 0
        self.cur_step = 0
        self.cur_patience = self.config.patience
        self.last_best = {}
        self.metrics_history = []

        self.train_dataloader = create_dataloader(
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            partition='training',
            batch_size=self.config.train_batch_size,
            subsample_ratio=self.config.train_subsample_ratio,
        )
        self.valid_dataloader = create_dataloader(
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            partition='validation',
            batch_size=self.config.valid_batch_size,
            subsample_ratio=self.config.valid_subsample_ratio,
        )

        _, s_true_wave = next(iter(self.train_dataloader))
        self.config.num_sources = s_true_wave.size(1)
        self.config.sample_length = s_true_wave.size(2)
        self.config.num_batches = len(self.train_dataloader)
        self.config.num_frequency_bins = 1 + self.config.stft_frame_size // 2

        if self.config.model_load_path:
            model = Model.load(
                path=os.path.join(self.config.model_load_path, BEST_CHECKPOINT_FILENAME),
                device=self.config.device
            ).train()
        else:
            model = None

        trainer_cls = get_trainer(self.config.model_name)
        self.trainer = trainer_cls(config=self.config, model=model)

        self.logger.info('model parameter count: %d', total_num_params(self.trainer.get_model().parameters()))
        self._train_loop()
        self.logger.info('completed')

    def _train_loop(self):
        while self.cur_patience > 0:
            for _ in range(self.config.eval_epochs):
                train_start = time()
                steps = self.trainer.train(self.train_dataloader)
                with tqdm(steps, total=self.config.num_batches, leave=False) as progress_bar:
                    for increment in steps:
                        self.cur_step += 1
                        progress_bar.update(increment)
                self.cur_epoch += 1

            metrics = {
                'process': {
                    'epoch': self.cur_epoch,
                    'step': self.cur_step,
                    'train_elapsed': time() - train_start,
                    'train_loss': self.trainer.get_loss()
                },
                'validation': {},
            }

            validate_start = time()
            with torch.inference_mode():
                metrics.update({'validation': self.trainer.validate(self.valid_dataloader)})
            metrics['process']['validate_elapsed'] = time() - validate_start

            self._update_best(metrics)
            self._update_patience(metrics)
            self._report(metrics)

    def _update_best(self, metrics):
        metrics['best'] = {}
        for key, value in metrics['validation'].items():
            if key not in self.last_best or value > self.last_best[key]:
                self.last_best[key] = value
                metrics['best'][key] = True
            else:
                metrics['best'][key] = False
        return metrics

    def _update_patience(self, metrics):
        min_epoch = self.config.min_epochs and self.config.min_epochs > self.cur_epoch

        if self.config.eval_method == 'reference':
            main_metric_name = 'sisnri'
        elif self.config.eval_method == 'blind':
            main_metric_name = 'sisnri_blind'
        else:
            raise Exception(f'unknown eval_method: {self.config.eval_method}')

        if min_epoch or metrics['best'][main_metric_name] or metrics['best'].get('sisnri_mixit_oracle'):
            self.cur_patience = self.config.patience
            self.trainer.get_model().save(os.path.join(self.config.results_dir, BEST_CHECKPOINT_FILENAME))
        else:
            self.cur_patience -= 1
        metrics['process']['patience'] = self.cur_patience

    def _report(self, metrics):
        log_line = ''
        for group in ['process', 'validation']:
            for key, value in metrics[group].items():
                format_spec = '{}={:.3f}' if isinstance(value, float) else '{}={}'
                log_line += format_spec.format(key, value)

                if group == 'validation' and metrics['best'][key]:
                    log_line += '*'
                else:
                    log_line += ' '

                if group == 'validation':
                    log_line += ' '

                self.tensorboard.add_scalar(
                    tag='{}/{}'.format(group, key),
                    scalar_value=value,
                    global_step=metrics['process']['step']
                )
            if group == 'process':
                log_line += '| '

        self.logger.info(log_line)

        self.tensorboard.flush()

        metrics['time'] = time()
        self.metrics_history.append(metrics)
        metrics_history_path = os.path.join(self.config.results_dir, METRICS_HISTORY_FILENAME)
        os.makedirs(os.path.dirname(metrics_history_path), exist_ok=True)
        torch.save(self.metrics_history, metrics_history_path)


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-results-root', type=str, required=True)
    arg_parser.add_argument('--librimix-root', type=str)
    arg_parser.add_argument('--realm-root', type=str)
    arg_parser.add_argument('--stft-frame-size', type=int)
    arg_parser.add_argument('--stft-hop-size', type=int)
    arg_parser.add_argument('--model-name', choices=['pit', 'pit-dm', 'mixit', 'mixpit', 'mixcycle'])
    arg_parser.add_argument('--mixcycle-init-epochs', type=int)
    arg_parser.add_argument('--snr-max', type=float)
    arg_parser.add_argument('--train-batch-size', type=int)
    arg_parser.add_argument('--valid-batch-size', type=int)
    arg_parser.add_argument('--lr', type=float)
    arg_parser.add_argument('--grad-clip', type=float)
    arg_parser.add_argument('--train-subsample-ratio', type=float)
    arg_parser.add_argument('--valid-subsample-ratio', type=float)
    arg_parser.add_argument('--eval-method', choices=['reference', 'blind'])
    arg_parser.add_argument('--eval-epochs', type=int)
    arg_parser.add_argument('--patience', type=int)
    arg_parser.add_argument('--seed', type=int)
    arg_parser.add_argument('--run-id', type=str)
    arg_parser.add_argument('--device-name', type=str)

    cmd_args = arg_parser.parse_args()
    Training(**vars(cmd_args)).start()
