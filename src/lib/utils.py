import os
import sys
import logging
import random

import torch


CONFIG_FILENAME = 'config.pth'
BEST_CHECKPOINT_FILENAME = 'best_checkpoint.pth'
BEST_METRICS_FILENAME = 'best_metrics.pth'
METRICS_HISTORY_FILENAME = 'metrics_history.pth'
EPS = 1e-8


class ResultsDirExistsError(Exception):
    pass


def configure_console_logger():
    logger = logging.getLogger()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s %(name)-6s %(levelname)-6s %(message)s', '%y-%m-%d %H:%M:%S'))
    logger.addHandler(console)


def configure_file_logger(work_dir):
    logger = logging.getLogger()
    file = logging.FileHandler(os.path.join(work_dir, 'root.log'))
    file.setFormatter(logging.Formatter('%(asctime)s %(name)-6s %(levelname)-6s %(message)s'))
    logger.addHandler(file)


def get_logger(name, work_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if work_dir:
        file = logging.FileHandler(os.path.join(work_dir, name + '.log'))
        file.setLevel(logging.DEBUG)
        file.setFormatter(logging.Formatter('%(asctime)s %(name)-6s %(levelname)-6s %(message)s'))
        logger.addHandler(file)
    return logger


def default(input_value, default_value):
    return default_value if input_value is None else input_value


def build_run_name(args, prepend_items=None, exclude_keys=None):
    if prepend_items is None:
        prepend_items = {}
    if exclude_keys is None:
        exclude_keys = []

    run_name_dict = {}
    if prepend_items is not None:
        run_name_dict.update(prepend_items)
    for k, v in args.items():
        if k not in prepend_items.keys() and k not in exclude_keys and v is not None:
            if isinstance(v, list):
                v = '_'.join(str(item) for item in v)
            run_name_dict[k] = v
    return '.'.join(f'{k}_{v}' for k, v in run_name_dict.items())


def ensure_clean_results_dir(results_dir):
    if os.path.exists(results_dir):
        raise ResultsDirExistsError(f'results dir "{results_dir}" exists, remove it and run the script again.')
    os.makedirs(results_dir, exist_ok=True)


def setup_determinism(seed):
    if seed is None:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def flatten_sources(batch):
    batch_size = batch.size(0)
    num_sources = batch.size(1)
    return batch.reshape(batch_size * num_sources, *batch.size()[2:])


def unflatten_sources(batch, num_sources):
    batch_size = batch.size(0) // num_sources
    return batch.view(batch_size, num_sources, *batch.size()[1:])


def shuffle_sources(batch, num_sources, shuffle_idx):
    batch_flat = flatten_sources(batch)
    return unflatten_sources(batch_flat[shuffle_idx], num_sources)


def total_num_params(params):
    return sum(param.numel() for param in params)


def soft_mask(m_pred_mag, x_true_mag):
    return (m_pred_mag / (m_pred_mag.sum(dim=1, keepdim=True) + EPS)) * x_true_mag


def metrics_to_str(metrics):
    return ' '.join('{}={:.3f}'.format(k, v) for k, v in metrics.items())


class MetricAccumulator:
    def __init__(self):
        self.accumulator = None
        self.reset()

    def store(self, batch):
        self.accumulator.append(batch.detach())

    def reset(self):
        self.accumulator = []

    def std_mean(self):
        with torch.inference_mode():
            epoch_values = torch.cat(self.accumulator)
            return torch.std_mean(epoch_values, unbiased=True)
