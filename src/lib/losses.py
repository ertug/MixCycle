from math import prod

import torch


def snr(true_wave, pred_wave, snr_max=None):
    true_wave_square_sum = true_wave.square().sum(-1)
    if snr_max is None:
        soft_threshold = 0
    else:
        threshold = 10 ** (-snr_max / 10)
        soft_threshold = threshold * true_wave_square_sum

    return 10 * torch.log10(true_wave_square_sum / ((true_wave - pred_wave).square().sum(-1) + soft_threshold))


def sisnr(true_wave, pred_wave, eps=0.):
    true_wave = ((true_wave * pred_wave).sum(-1) / true_wave.square().sum(-1)).unsqueeze(-1) * true_wave
    return 10 * torch.log10(true_wave.square().sum(-1) / ((true_wave - pred_wave).square().sum(-1) + eps))


def sisnri(true_wave, pred_wave, x_true_wave, eps=0.):
    return sisnr(true_wave=true_wave, pred_wave=pred_wave, eps=eps) - \
           sisnr(true_wave=true_wave, pred_wave=x_true_wave, eps=eps)


def negative_snr(true_wave, pred_wave, snr_max=None):
    return -snr(true_wave, pred_wave, snr_max)


def negative_sisnr(true_wave, pred_wave):
    return -sisnr(true_wave, pred_wave)


def negative_sisnri(true_wave, pred_wave, x_true_wave, eps=0.):
    return -sisnri(true_wave, pred_wave, x_true_wave, eps=eps)


def invariant_loss(true, pred, mixing_matrices, loss_func, return_best_perm_idx=False):
    pred_flat = pred.view(*pred.size()[:2], prod(pred.size()[2:]))

    batch_size = true.size(0)
    perm_size = mixing_matrices.size(0)
    loss_perms = torch.empty([batch_size, perm_size], device=true.device)

    for perm_idx in range(perm_size):
        pred_flat_mix = mixing_matrices[perm_idx].matmul(pred_flat)
        pred_mix = pred_flat_mix.view(*pred_flat_mix.size()[:2], *pred.size()[2:])
        loss_perms[:, perm_idx] = loss_func(true, pred_mix).mean(dim=1)
    _, best_perm_idx = loss_perms.min(dim=1)

    batch_loss = loss_perms[torch.arange(batch_size), best_perm_idx]

    if return_best_perm_idx:
        return batch_loss, best_perm_idx
    else:
        return batch_loss
