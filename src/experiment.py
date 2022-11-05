import os
import argparse
from types import SimpleNamespace

import torch

from lib.utils import CONFIG_FILENAME, configure_console_logger, configure_file_logger, get_logger
from train import Training
from test import Test
from irm import IdealRatioMask


class Experiment:
    def __init__(self, experiments_root, experiment_name, librimix_root=None, realm_root=None):
        self.config = SimpleNamespace()
        self.config.experiments_root = experiments_root
        self.config.datasets_root = os.path.join(experiments_root, 'datasets')
        self.config.checkpoints_root = os.path.join(experiments_root, experiment_name)
        self.config.librimix_root = librimix_root
        self.config.realm_root = realm_root

        if os.path.exists(self.config.checkpoints_root):
            raise Exception(f'checkpoints root "{self.config.checkpoints_root}" exists!')
        os.makedirs(self.config.checkpoints_root, exist_ok=True)

        configure_file_logger(self.config.checkpoints_root)
        self.logger = get_logger('exp', self.config.checkpoints_root)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.checkpoints_root, CONFIG_FILENAME))

    def librimix_irm(self):
        IdealRatioMask(
            irm_results_root=self.config.checkpoints_root,
            librimix_root=self.config.librimix_root
        ).start()

    def model_comparison_librimix(self,
                                  model_names=None,
                                  mixcycle_init_epochs=50,
                                  train_subsample_ratio=1.,
                                  eval_epochs=1,
                                  run_id=None):
        all_model_names = ['mixcycle', 'pit', 'pit-dm', 'mixit', 'mixpit']
        for model_name in (all_model_names if model_names is None else model_names):
            self.logger.info('model_name=%s', model_name)

            training = Training(
                train_results_root=self.config.checkpoints_root,
                librimix_root=self.config.librimix_root,
                model_name=model_name,
                mixcycle_init_epochs=mixcycle_init_epochs,
                train_subsample_ratio=train_subsample_ratio,
                eval_epochs=eval_epochs,
                run_id=run_id
            )
            training.start()

            Test(
                train_results_dir=training.results_dir,
                librimix_root=self.config.librimix_root
            ).start()

    def librimix_5p(self):
        self.model_comparison_librimix(
            #model_names=['mixcycle'],
            mixcycle_init_epochs=250,
            train_subsample_ratio=0.05,
            eval_epochs=10
        )

    def librimix_100p(self):
        self.model_comparison_librimix(
            mixcycle_init_epochs=50,
            train_subsample_ratio=1.,
            eval_epochs=1
        )

    def realm(self):
        initialized_model = Training(
            train_results_root=os.path.join(self.config.experiments_root, 'librimix_100p'),
            librimix_root=True,
            model_name='mixcycle',
            mixcycle_init_epochs=50,
            train_subsample_ratio=1.0,
            eval_epochs=1
        )

        training = Training(
            train_results_root=self.config.checkpoints_root,
            realm_root=self.config.realm_root,
            model_name='mixcycle',
            model_load_path=initialized_model.results_dir,
            mixcycle_init_epochs=0,
            eval_method='blind',
            eval_blind_num_repeat=20,
            eval_epochs=40
        )
        training.start()

        Test(
            train_results_dir=training.results_dir,
            realm_root=self.config.realm_root,
            eval_blind_num_repeat=100
        ).start()


if __name__ == '__main__':
    configure_console_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-root', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--librimix-root', type=str)
    parser.add_argument('--realm-root', type=str)

    args = parser.parse_args()
    experiment = Experiment(
        experiments_root=args.exp_root,
        experiment_name=args.run,
        librimix_root=args.librimix_root,
        realm_root=args.realm_root
    )

    try:
        func = getattr(experiment, args.run)
    except AttributeError:
        raise Exception('unknown experiment')

    func()
