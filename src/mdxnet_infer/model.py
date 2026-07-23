# Copyright (c) 2023 KUIELab (sdx23 TFC-TDF v3 architecture)
# Copyright (c) 2023 Roman Solovyev / ZFTurbo (Music-Source-Separation-Training)
# SPDX-License-Identifier: MIT

"""TFC-TDF v3 encoder-decoder network: the MDX23C architecture used for drum
stem separation.

Verbatim inference-only port of ZFTurbo's Music-Source-Separation-Training
`models/mdx23c_tfc_tdf_v3.py` (itself based on KUIELab's sdx23 TFC-TDF v3),
with training-only code stripped and the config plumbing adapted to this
package's dataclasses. `TFC_TDF_net.forward()` is the model referenced by
`MDX23CInference` — it does STFT -> subband reshape -> U-Net-style TFC-TDF
encoder/decoder -> masked reconstruction -> inverse STFT, producing one
output channel group per stem.

Reads: .config
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Tuple

from .config import MDX23CConfig


class STFT:
    """Short-Time Fourier Transform for audio processing."""

    def __init__(self, n_fft: int, hop_length: int, dim_f: int, device: torch.device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply STFT to audio tensor."""
        # Handle MPS device by moving to CPU
        x_is_mps = x.device.type not in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            window=window, center=True, return_complex=False
        )
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape(
            [*batch_dims, c * 2, -1, x.shape[-1]]
        )

        if x_is_mps:
            x = x.to(self.device)

        return x[..., :self.dim_f, :]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse STFT to spectrogram tensor."""
        x_is_mps = x.device.type not in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])

        if x_is_mps:
            x = x.to(self.device)

        return x


def get_norm(norm_type: str):
    """Get normalization layer factory."""
    def norm(c: int, norm_type: str):
        if norm_type is None:
            return nn.Identity()
        elif norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


def get_act(act_type: str) -> nn.Module:
    """Get activation function."""
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type.startswith('elu'):
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


class Upscale(nn.Module):
    """Upscaling block with transposed convolution."""

    def __init__(self, in_c: int, out_c: int, scale: Tuple[int, int], norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(
                in_channels=in_c, out_channels=out_c,
                kernel_size=scale, stride=scale, bias=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downscale(nn.Module):
    """Downscaling block with strided convolution."""

    def __init__(self, in_c: int, out_c: int, scale: Tuple[int, int], norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c,
                kernel_size=scale, stride=scale, bias=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TFC_TDF(nn.Module):
    """
    Time-Frequency Convolution with Time-Distributed Fully-connected layers.

    Core building block of the MDX23C architecture.
    """

    def __init__(self, in_c: int, c: int, num_blocks: int, f: int, bn: int, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    """
    TFC-TDF encoder-decoder network for audio source separation.

    This is the main MDX23C architecture used for drum stem separation.

    Args:
        config: MDX23CConfig with model and audio parameters
        device: torch.device for the model
    """

    def __init__(self, config: MDX23CConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        norm = get_norm(norm_type=config.model.norm)
        act = get_act(act_type=config.model.act)

        # Number of target stems
        self.num_target_instruments = (
            1 if config.training.target_instrument
            else len(config.training.instruments)
        )
        self.num_subbands = config.model.num_subbands

        # Input dimension
        dim_c = self.num_subbands * config.audio.num_channels * 2
        n = config.model.num_scales
        scale = tuple(config.model.scale)
        num_blocks_per_scale = config.model.num_blocks_per_scale
        c = config.model.num_channels
        g = config.model.growth
        bn = config.model.bottleneck_factor
        f = config.audio.dim_f // self.num_subbands

        # First convolution
        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, num_blocks_per_scale, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        # Bottleneck
        self.bottleneck_block = TFC_TDF(c, c, num_blocks_per_scale, f, bn, norm, act)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, num_blocks_per_scale, f, bn, norm, act)
            self.decoder_blocks.append(block)

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        # STFT processor
        self.stft = STFT(
            config.audio.n_fft,
            config.audio.hop_length,
            config.audio.dim_f,
            self.device
        )

    def cac2cws(self, x: torch.Tensor) -> torch.Tensor:
        """Convert channel-as-channel to channel-with-subbands format."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x: torch.Tensor) -> torch.Tensor:
        """Convert channel-with-subbands to channel-as-channel format."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Audio tensor of shape (batch, channels, samples)

        Returns:
            Separated stems of shape (batch, num_stems, channels, samples)
            or (batch, channels, samples) if single stem
        """
        # Apply STFT
        x = self.stft(x)

        # Convert to subband format
        mix = x = self.cac2cws(x)

        # First convolution
        first_conv_out = x = self.first_conv(x)

        # Transpose for TDF
        x = x.transpose(-1, -2)

        # Encoder
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        # Bottleneck
        x = self.bottleneck_block(x)

        # Decoder
        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        # Transpose back
        x = x.transpose(-1, -2)

        # Apply learned mask
        x = x * first_conv_out

        # Final convolution
        x = self.final_conv(torch.cat([mix, x], 1))

        # Convert back from subband format
        x = self.cws2cac(x)

        # Reshape for multi-stem output
        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        # Inverse STFT
        x = self.stft.inverse(x)

        return x
