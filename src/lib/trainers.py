import random
from math import prod
from functools import partial

import torch

from lib.losses import negative_snr, negative_sisnri, invariant_loss
from lib.models import Model
from lib.utils import EPS, shuffle_sources, MetricAccumulator
from lib.data.dataloader_utils import double_mixture_generator


class Trainer:
    def __init__(self, config=None, model=None):
        self.loss_accumulator = MetricAccumulator()
        self.sisnri_accumulator = MetricAccumulator()
        if model:
            self.model = model
            if config:
                self.model.config = config
        else:
            self.model = Model(config=config)
        self.model.to(self.model.config.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.model.config.lr},
        ])

    def train(self, dataloader):
        self.model.train()
        self.loss_accumulator.reset()

    def step(self, batch_loss):
        self.optimizer.zero_grad(set_to_none=True)
        batch_loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model.config.grad_clip)
        self.optimizer.step()
        self.loss_accumulator.store(batch_loss)

    def validate(self, dataloader):
        self.model.eval()
        self.sisnri_accumulator.reset()

        for x_true_wave, s_true_wave in dataloader:
            x_true_wave = x_true_wave.to(self.model.config.device)
            s_true_wave = s_true_wave.to(self.model.config.device)

            s_pred_wave = self.model(x_true_wave)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.num_sources,
                num_mix=1,
                allow_empty=True
            )
            batch_sisnri = -invariant_loss(
                true=s_true_wave,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_sisnri, x_true_wave=x_true_wave)
            )
            self.sisnri_accumulator.store(batch_sisnri)

        std, mean = self.sisnri_accumulator.std_mean()
        return {
            'sisnri': mean.item(),
            'sisnri_std': std.item()
        }

    def get_loss(self):
        _, mean = self.loss_accumulator.std_mean()
        return mean.item()

    def get_model(self):
        return self.model


class PermutationInvariantTrainer(Trainer):
    def train(self, dataloader):
        super().train(dataloader)

        for x_true_wave, s_true_wave in dataloader:
            x_true_wave = x_true_wave.to(self.model.config.device)
            s_true_wave = s_true_wave.to(self.model.config.device)

            s_pred_wave = self.model(x_true_wave)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.config.num_sources,
                num_mix=1
            )
            batch_loss = invariant_loss(
                true=s_true_wave,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_snr, snr_max=self.model.config.snr_max),
            )
            self.step(batch_loss)
            yield 1


class PermutationInvariantDynamicMixingTrainer(Trainer):
    def train(self, dataloader):
        super().train(dataloader)

        for _, s_true_wave in dataloader:
            s_true_wave = s_true_wave.to(self.model.config.device)

            shuffle_idx = torch.randperm(prod(s_true_wave.size()[:2]))
            s_true_wave_shuffled = shuffle_sources(s_true_wave, self.model.config.num_sources, shuffle_idx)
            x_true_wave_shuffled = s_true_wave_shuffled.sum(dim=1, keepdims=True)

            s_pred_wave_shuffled = self.model(x_true_wave_shuffled)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.config.num_sources,
                num_mix=1
            )
            batch_loss = invariant_loss(
                true=s_true_wave_shuffled,
                pred=s_pred_wave_shuffled,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_snr, snr_max=self.model.config.snr_max),
            )
            self.step(batch_loss)
            yield 1


class MixtureInvariantTrainer(Trainer):
    def __init__(self, config=None, model=None):
        if not model:
            model = Model(config=config, num_sources=config.num_sources * 2)

        super().__init__(config, model)

        self.sisnri_mixit_oracle_accumulator = MetricAccumulator()

    def train(self, dataloader):
        super().train(dataloader)

        for x_true_wave_1, x_true_wave_2 in double_mixture_generator(dataloader):
            x_true_wave_1 = x_true_wave_1.to(self.model.config.device)
            x_true_wave_2 = x_true_wave_2.to(self.model.config.device)

            x_true_wave_double = torch.cat([x_true_wave_1, x_true_wave_2], dim=1)
            x_true_wave_mom = x_true_wave_double.sum(dim=1, keepdim=True)

            s_pred_wave = self.model(x_true_wave_mom)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=2,
                max_sources=self.model.num_sources
            )
            batch_loss = invariant_loss(
                true=x_true_wave_double,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_snr, snr_max=self.model.config.snr_max),
            )
            self.step(batch_loss)
            yield 2

    def validate(self, dataloader):
        metrics = super().validate(dataloader)

        self.sisnri_mixit_oracle_accumulator.reset()

        for x_true_wave, s_true_wave in dataloader:
            x_true_wave = x_true_wave.to(self.model.config.device)
            s_true_wave = s_true_wave.to(self.model.config.device)

            s_pred_wave = self.model(x_true_wave)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.num_sources
            )
            batch_sisnri = -invariant_loss(
                true=s_true_wave,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_sisnri, x_true_wave=x_true_wave, eps=EPS),
            )

            self.sisnri_mixit_oracle_accumulator.store(batch_sisnri)

        std, mean = self.sisnri_mixit_oracle_accumulator.std_mean()
        metrics['sisnri_mixit_oracle'] = mean.item()
        metrics['sisnri_mixit_oracle_std'] = std.item()
        return metrics


class MixturePermutationInvariantTrainer(Trainer):
    def train(self, dataloader):
        super().train(dataloader)

        for x_true_wave_1, x_true_wave_2 in double_mixture_generator(dataloader):
            x_true_wave_1 = x_true_wave_1.to(self.model.config.device)
            x_true_wave_2 = x_true_wave_2.to(self.model.config.device)

            x_true_wave_double = torch.cat([x_true_wave_1, x_true_wave_2], dim=1)
            x_true_wave_mom = x_true_wave_double.sum(dim=1, keepdim=True)

            s_pred_wave = self.model(x_true_wave_mom)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.config.num_sources,
                num_mix=1
            )
            batch_loss = invariant_loss(
                true=x_true_wave_double,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_snr, snr_max=self.model.config.snr_max),
            )
            self.step(batch_loss)
            yield 2


class MixCycleTrainer(Trainer):
    def __init__(self, config=None, model=None):
        super().__init__(config, model)

        self.sisnri_blind_accumulator = MetricAccumulator()
        self.mixpit_trainer = MixturePermutationInvariantTrainer(config, self.model)
        self.epochs = 0
        self.mixcycle_steps = 0
        self.model_copy = None

    def train(self, dataloader):
        super().train(dataloader)

        if self.epochs < self.model.config.mixcycle_init_epochs:
            generator = self.mixpit_trainer.train(dataloader)
        else:
            self.mixpit_trainer = None
            generator = self._mixcycle_train(dataloader)

        self.epochs += 1
        return generator

    def validate(self, dataloader):
        if self.mixpit_trainer:
            metrics = self.mixpit_trainer.validate(dataloader)
        else:
            metrics = super().validate(dataloader)

        if self.model.config.eval_method == 'blind':
            metrics.update(self.validate_blind(dataloader))
        return metrics

    def validate_blind(self, dataloader):
        self.sisnri_blind_accumulator.reset()

        for _ in range(self.model.config.eval_blind_num_repeat):
            for x_true_wave, _ in dataloader:
                x_true_wave = x_true_wave.to(self.model.config.device)

                x_pred_wave_shuffled, s_pred_wave_shuffled = self._teacher(x_true_wave)

                s_pred_wave = self.model(x_pred_wave_shuffled)

                mixing_matrices = self.model.generate_mixing_matrices(
                    num_targets=self.model.config.num_sources,
                    max_sources=self.model.config.num_sources,
                    num_mix=1
                )
                batch_sisnri = -invariant_loss(
                    true=s_pred_wave_shuffled,
                    pred=s_pred_wave,
                    mixing_matrices=mixing_matrices,
                    loss_func=partial(negative_sisnri, x_true_wave=x_pred_wave_shuffled, eps=EPS),
                )

                self.sisnri_blind_accumulator.store(batch_sisnri)

        std, mean = self.sisnri_blind_accumulator.std_mean()
        return {
            'sisnri_blind': mean.item(),
            'sisnri_blind_std': std.item()
        }

    def get_loss(self):
        if self.mixpit_trainer:
            return self.mixpit_trainer.get_loss()
        else:
            return super().get_loss()

    def get_model(self):
        if self.mixpit_trainer:
            return self.mixpit_trainer.get_model()
        else:
            return super().get_model()

    def _mixcycle_train(self, dataloader):
        for x_true_wave, _ in dataloader:
            with torch.no_grad():
                x_true_wave = x_true_wave.to(self.model.config.device)

                x_pred_wave_shuffled, s_pred_wave_shuffled = self._teacher(x_true_wave)

            s_pred_wave = self.model(x_pred_wave_shuffled)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.config.num_sources,
                num_mix=1
            )
            batch_loss = invariant_loss(
                true=s_pred_wave_shuffled,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_snr, snr_max=self.model.config.snr_max),
            )
            self.step(batch_loss)
            self.mixcycle_steps += 1
            yield 1

    def _teacher(self, x_true_wave):
        s_pred_wave = self.model(x_true_wave)

        shuffle_idx = list(range(prod(s_pred_wave.size()[:2])))
        for i in range(0, len(shuffle_idx), 2):
            # randomly swap source estimates of mixture
            if random.randrange(2) == 0:
                shuffle_idx[i], shuffle_idx[i + 1] = shuffle_idx[i + 1], shuffle_idx[i]

        for i in range(0, len(shuffle_idx), 4):
            # swap source estimates across two mixtures
            shuffle_idx[i], shuffle_idx[i + 2] = shuffle_idx[i + 2], shuffle_idx[i]

        s_pred_wave_shuffled = shuffle_sources(s_pred_wave, self.model.config.num_sources, shuffle_idx)
        x_pred_wave_shuffled = s_pred_wave_shuffled.sum(dim=1, keepdims=True)

        return x_pred_wave_shuffled, s_pred_wave_shuffled


TRAINER_MAPPING = {
    'pit': PermutationInvariantTrainer,
    'pit-dm': PermutationInvariantDynamicMixingTrainer,
    'mixit': MixtureInvariantTrainer,
    'mixpit': MixturePermutationInvariantTrainer,
    'mixcycle': MixCycleTrainer,
}


def get_trainer(model_name):
    try:
        return TRAINER_MAPPING[model_name]
    except KeyError:
        raise Exception(f'unknown model name: {model_name}')
