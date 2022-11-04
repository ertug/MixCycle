import torch

from lib.utils import flatten_sources, unflatten_sources


class Transform:
    def __init__(self, stft_frame_size, stft_hop_size, device):
        self.stft_frame_size = stft_frame_size
        self.stft_hop_size = stft_hop_size

        self.hann_window = torch.hann_window(
            self.stft_frame_size,
            periodic=True,
            device=device
        )

    def stft(self, wave):
        wave_flat = flatten_sources(wave)
        complex_flat = torch.stft(
            wave_flat,
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            return_complex=True
        )
        complex = unflatten_sources(complex_flat, num_sources=wave.size(1))
        mag, phase = complex.abs(), complex.angle()
        return mag, phase

    def istft(self, mag, phase, length):
        complex = torch.complex(
            real=mag * phase.cos(),
            imag=mag * phase.sin()
        )
        complex_flat = flatten_sources(complex)
        wave_flat = torch.istft(
            complex_flat,
            n_fft=self.stft_frame_size,
            hop_length=self.stft_hop_size,
            window=self.hann_window,
            length=length
        )
        return unflatten_sources(wave_flat, num_sources=mag.size(1))
