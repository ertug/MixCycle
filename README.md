# MixCycle: Unsupervised Speech Separation via Cyclic Mixture Permutation Invariant Training
This repository contains the audio samples and the source code that accompany [the paper](https://arxiv.org/abs/2202.03875).

## Audio samples
We provide audio samples to demonstrate the results of the MixCycle method on two different datasets: [LibriMix](https://nbviewer.org/github/ertug/MixCycle/blob/main/notebooks/AudioSamples-LibriMix.ipynb) and [REAL-M](https://nbviewer.org/github/ertug/MixCycle/blob/main/notebooks/AudioSamples-REAL-M.ipynb).

Also note that the provided [REAL-M](https://nbviewer.org/github/ertug/MixCycle/blob/main/notebooks/AudioSamples-REAL-M.ipynb) samples were used in the informal listening test.

We also provide audio samples from the baseline methods on LibriMix: [PIT-DM](https://nbviewer.org/github/ertug/MixCycle/blob/main/notebooks/AudioSamples-LibriMix-PIT-DM.ipynb) and [MixIT](https://nbviewer.org/github/ertug/MixCycle/blob/main/notebooks/AudioSamples-LibriMix-MixIT.ipynb).

## Source code
We provide the source code under the `src` directory for reproducibility.

## Running the experiments

### Prepare the datasets
- LibriMix: [GitHub](https://github.com/JorisCos/LibriMix)
- REAL-M: [Download](https://sourceseparationresearch.com/static/REAL-M-v0.1.0.tar.gz)

### Create the environment

Install [Anaconda](https://www.anaconda.com/products/individual) and run the following command:
```
$ conda env create -f environment.yml
```
See [more info](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on how to manage conda environments.

### Activate the environment
```
$ conda activate mixcycle
```

### Run the experiments
```
$ cd src
$ python experiment.py --librimix-root ~/datasets/librimix --exp-root ~/experiments --run librimix_irm
$ python experiment.py --librimix-root ~/datasets/librimix --exp-root ~/experiments --run librimix_5p
$ python experiment.py --librimix-root ~/datasets/librimix --exp-root ~/experiments --run librimix_100p
$ python experiment.py --librimix-root ~/datasets/librimix --realm-root ~/datasets/REAL-M-v0.1.0 --exp-root ~/experiments --run realm
```

Optionally, you can monitor the training process with TensorBoard by running:
```
$ tensorboard --logdir experiments
```

## Citation (BibTeX)
If you find this repository useful, please cite our work:

```BibTeX
@article{karamatli2022unsupervised,
  title={MixCycle: Unsupervised Speech Separation via Cyclic Mixture Permutation Invariant Training},
  author={Karamatl{\i}, Ertu{\u{g}} and K{\i}rb{\i}z, Serap},
  journal={IEEE Signal Processing Letters},
  volume={29},
  number={},
  pages={2637-2641},
  year={2022},
  doi={10.1109/LSP.2022.3232276}
}
```
