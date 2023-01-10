import os
import sys
from functools import partial
from types import SimpleNamespace

import torch
from torchaudio.datasets.librimix import LibriMix
import ipywidgets as widgets
import IPython.display as ipd

from train import BEST_CHECKPOINT_FILENAME, Training
from lib.data.realm import RealM
from lib.models import Model
from lib.losses import negative_sisnri, invariant_loss


SAMPLE_RATE = 8000


def collate_samples(dataset_name, dataset_root, count):
    if dataset_name == 'librimix':
        dataset = LibriMix(
            root=dataset_root,
            subset='test',
            num_speakers=2,
            sample_rate=SAMPLE_RATE,
            task='sep_clean',
        )
    elif dataset_name == 'realm':
        dataset = RealM(
            root=dataset_root,
            partition='testing',
        )
    else:
        raise Exception(f'unknown dataset name: {dataset_name}')

    selected_idxs = torch.randperm(len(dataset))[:count]
    data_samples = []
    for idx in selected_idxs:
        _, x_true_wave, s_true_wave = dataset[idx]
        data_sample = SimpleNamespace(
            x_true_wave=x_true_wave,
            s_true_wave=torch.cat(s_true_wave)
        )
        data_samples.append(data_sample)
    return data_samples


def evaluate_samples(train_results_root, dataset_name, model_name, data_samples, **kwargs):
    training = Training(
        train_results_root=train_results_root,
        librimix_root=(dataset_name == 'librimix'),
        realm_root=(dataset_name == 'realm'),
        model_name=model_name,
        **kwargs
    )

    with torch.inference_mode():
        model = Model.load(
            path=os.path.join(training.results_dir, BEST_CHECKPOINT_FILENAME),
            device='cpu'
        ).eval()

        separation_samples = []
        for data_sample in data_samples:
            s_pred_wave = model(data_sample.x_true_wave.unsqueeze(0))
            mixing_matrices = model.generate_mixing_matrices(
                num_targets=model.config.num_sources,
                max_sources=model.num_sources,
                num_mix=1,
                allow_empty=True
            )
            negative_sisnri_value, best_perm_idx = invariant_loss(
                true=data_sample.s_true_wave.unsqueeze(0),
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_sisnri, x_true_wave=data_sample.x_true_wave.unsqueeze(0)),
                return_best_perm_idx=True
            )
            s_pred_wave_permuted = mixing_matrices[best_perm_idx].matmul(s_pred_wave)

            separation_samples.append(SimpleNamespace(
                x_true_wave=data_sample.x_true_wave,
                s_true_wave=data_sample.s_true_wave,
                s_pred_wave=s_pred_wave_permuted.squeeze(0),
                sisnri=-negative_sisnri_value.item()
            ))
        return separation_samples


def _audio_widget(data):
    out = widgets.Output()
    with out:
         ipd.display(ipd.Audio(data, rate=SAMPLE_RATE))
    return out


def _mixture_widget(html, sample):
    return widgets.VBox([
        widgets.HTML(html),
        _audio_widget(sample.x_true_wave)
    ])


def show_samples_librimix(separation_samples):
    def _source_widget(text, s_wave):
        return widgets.VBox([
            widgets.HTML('<h3>{} 1</h3>'.format(text)),
            _audio_widget(s_wave[0]),
            widgets.HTML('<h3>{} 2</h3>'.format(text)),
            _audio_widget(s_wave[1]),
        ])

    def sisnri(separation_sample):
        html = '<div style="text-align: center; white-space: nowrap;">Mean SI-SNRi:</div>' \
               '<div style="text-align: center; font-weight:bold; white-space: nowrap;">{:.1f}</div>' \
               .format(separation_sample.sisnri)
        return widgets.HTML(html)

    for idx, separation_sample in enumerate(separation_samples):
        ipd.display(widgets.HBox([
            _mixture_widget(html='<h2>Mixture #{}</h2>'.format(idx+1), sample=separation_sample),
            _source_widget(text='Reference Source', s_wave=separation_sample.s_true_wave),
            _source_widget(text='Estimated Source', s_wave=separation_sample.s_pred_wave),
            sisnri(separation_sample),
        ], layout=widgets.Layout(align_items='center', padding='3px', margin='5px', border='2px solid gray')))


def show_samples_realm(separation_samples_librimix, separation_samples_realm):
    def _source_widget(model_id, sample):
        return widgets.VBox([
            widgets.HTML('<h3>Model {} / Est. Source 1</h3>'.format(model_id)),
            _audio_widget(sample.s_pred_wave[0]),
            widgets.HTML('<h3>Model {} / Est. Source 2</h3>'.format(model_id)),
            _audio_widget(sample.s_pred_wave[1]),
        ])

    for idx, (sample_librimix, sample_realm) in enumerate(zip(separation_samples_librimix, separation_samples_realm)):
        ipd.display(widgets.HBox([
            _mixture_widget(html='<h2>Mixture #{}</h2>'.format(idx+1), sample=sample_librimix),
            _source_widget(model_id='A', sample=sample_librimix),
            _source_widget(model_id='B', sample=sample_realm)
        ], layout=widgets.Layout(align_items='center', padding='3px', margin='5px', border='2px solid gray')))


def load_file(path):
    try:
        data = torch.load(path)
        return data
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return None
