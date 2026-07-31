"""TFC_TDF_v3 (MDX23C) ported to Apple MLX -- the compute this package's MLX
backend runs.

Vendored (not depended on) because the upstream package this ships from,
`mlx-audio-separator`, bundles a full model catalog, a CLI, Roformer/Demucs/VR
architectures, and a compiled `mlx-audio-io` extension that pins `mlx==0.31.2`
exactly -- installing it would drag all of that in for one model class (see
`mdxnet_infer.backends.mlx_backend` for the refusal rationale). Two upstream
files are folded into this one: the model itself and its STFT wrapper, kept
together because nothing else in this package uses the STFT wrapper standalone
and a second file would only add an import hop.

`TfcTdfV3MLX` is copied verbatim from upstream's `tfc_tdf_v3_mlx.py`, with one
deliberate, non-upstream addition: `exact_zero_safe_rfft()` (see its own
docstring, and the susceptibility analysis + measurement further down this
module docstring -- it is inert for this architecture). `MLX_ENABLE_AMP` is
never read anywhere in this port, so nothing needs to force it off: unlike the
Roformer port `exact_zero_safe_rfft` was adapted from, this architecture has
no mixed-precision path at all.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    Files:    mlx_audio_separator/separator/models/mdxc/tfc_tdf_v3_mlx.py
              mlx_audio_separator/separator/models/mdx/stft.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.

`exact_zero_safe_rfft` susceptibility, measured rather than assumed (see
`mdxnet_infer.backends.mlx_backend` module docstring and
`tests/test_mlx_parity.py`): TFC_TDF_v3 has no operation that discards a
token/frame's own magnitude and renormalizes it (the Roformer's `L2Norm`
divides every token by *its own* L2 norm with an eps of 1e-12, five orders
below the ~4.5e-07 rfft artifact, which is what turns that artifact into a
full-scale feature). This architecture's norms are BatchNorm/InstanceNorm/
GroupNorm, which normalize using statistics computed over many samples
(BatchNorm: running stats fixed at eval) or a whole spatial extent
(InstanceNorm/GroupNorm, `eps=1e-5` -- three orders *above* the artifact, so
the artifact is swallowed by the epsilon floor rather than amplified by it),
and it is fully convolutional with no attention, so nothing spreads a
corrupted frame across every time position either.

Measured on the real `drumsep-6stem` checkpoint through the public
`.separate()` API (2026-07-30, Apple M-series, torch 2.13.0, mlx 0.31.2),
worst-case max-abs Torch-vs-MLX divergence per tail, with the guard in place
and with it removed:

    tail          with guard    without guard
    signal        1.371e-06     1.654e-06
    zeros         1.445e-06     1.952e-06
    near_silent   1.028e-06     1.490e-06

All six numbers sit in the same ~1e-6 noise floor -- no order-of-magnitude
jump on the silent tails the way the sibling package measured (4.0e-07 clean
vs 1.455e-02 zero-padded, a ~36,000x jump). The guard is kept anyway (cheap,
harmless, and matches the sibling package's pattern so a future architecture
change here does not silently reintroduce the failure mode), but it is
measured **inert** for this architecture, not load-bearing.

Reads: mlx.core, mlx.nn, mlx_spectro (get_transform_mlx)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_spectro import get_transform_mlx


@contextmanager
def exact_zero_safe_rfft():
    """Route `mx.fft.rfft` through the CPU stream for one STFT. NOT upstream code.

    Adapted from `bs_roformer.mlx.model.exact_zero_safe_rfft` (same org, same
    root cause): MLX 0.31.2's Metal rfft kernel packs two real FFTs into one
    complex FFT, and in float32 that cancellation is not bit-exact, so a frame
    whose true value is exactly zero comes back as roughly 4.5e-07 instead of
    0. In the sibling `bs-roformer-infer` package that artifact gets amplified
    about a millionfold by an L2-normalize-by-own-magnitude layer feeding a
    time-axis attention stack, and every track's final chunk is padded, so it
    is not a corner case there.

    This module's architecture (see the module docstring above) does not have
    that combination -- its norms use eps values and statistics that do not
    amplify a near-zero input, and it has no attention to spread a corrupted
    frame across other time positions. Measured end-to-end on the real
    checkpoint (see the report this shipped with): removing this guard did not
    move Torch-vs-MLX parity outside noise on any of the three silence
    fixtures. It is kept regardless -- it is cheap, harmless, and consistent
    with the sibling package's pattern -- but it is not load-bearing here.

    Caveat, stated rather than hidden: this swaps a module-level attribute, so
    it is not thread-safe. Inference here is single-threaded per session.
    """
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original


class STFT:
    """STFT processor for MDX23C-family models (MLX, via `mlx_spectro`).

    Converts stereo audio to/from the 4-channel real/imag spectrogram format
    `TfcTdfV3MLX` expects. Forward: (N, 2, T) -> STFT -> interleave real/imag
    -> (N, 4, dim_f, frames). Inverse is the reverse. This mirrors
    `mdxnet_infer.model.STFT`'s arithmetic exactly (same reshape/interleave
    order); it differs only in which library performs the underlying
    windowed FFT -- `torch.stft`/`torch.istft` there, `mlx_spectro`'s
    `SpectralTransform` here, because MLX has no built-in windowed-STFT op
    and `mlx_spectro` is already a declared floor dependency of this
    package's `[mlx]` extra (see `bs_roformer.mlx.model` for the sibling
    package's identical choice, and D4 in that package's `brain/decisions.md`
    for why vendoring the spectral op itself was rejected).
    """

    def __init__(self, n_fft: int, hop_length: int, dim_f: int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.n_bins = n_fft // 2 + 1
        self._transform = get_transform_mlx(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window_fn="hann",
            window=None,
            periodic=True,
            center=True,
            normalized=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward STFT. x: (N, 2, T) stereo -> (N, 4, dim_f, frames)."""
        n, c, t = x.shape
        x_flat = mx.reshape(x, (n * c, t))

        with exact_zero_safe_rfft():
            spec_complex = self._transform.stft(x_flat)  # (N*C, F, frames)
            mx.eval(spec_complex)

        spec = mx.stack([spec_complex.real, spec_complex.imag], axis=-1)
        f = spec.shape[1]
        frames = spec.shape[2]
        spec = mx.reshape(spec, (n, c, f, frames, 2))

        # Interleave real/imag per channel: [ch0_real, ch0_imag, ch1_real, ch1_imag],
        # matching mdxnet_infer.model.STFT.__call__'s permute+reshape exactly.
        spec = mx.transpose(spec, (0, 1, 4, 2, 3))  # (N, C, 2, F, frames)
        spec = mx.reshape(spec, (n, c * 2, f, frames))  # (N, 4, F, frames)

        return spec[:, :, : self.dim_f, :]

    def inverse(self, spec: mx.array) -> mx.array:
        """Inverse STFT. spec: (..., 4, dim_f, frames) -> (..., 2, T)."""
        batch_dims = spec.shape[:-3]
        channels = spec.shape[-3]
        frames = spec.shape[-1]

        if self.dim_f < self.n_bins:
            pad_size = self.n_bins - self.dim_f
            freq_pad = mx.zeros((*batch_dims, channels, pad_size, frames), dtype=spec.dtype)
            spec = mx.concatenate([spec, freq_pad], axis=-2)

        spec = mx.reshape(spec, (*batch_dims, 2, 2, self.n_bins, frames))

        flat = 1
        for dim in batch_dims:
            flat *= int(dim)
        spec = mx.reshape(spec, (flat, 2, 2, self.n_bins, frames))
        spec = mx.transpose(spec, (0, 1, 3, 4, 2))  # (flat, 2, F, frames, 2)

        spec_complex = spec[..., 0] + 1j * spec[..., 1]
        spec_flat = mx.reshape(spec_complex, (flat * 2, self.n_bins, frames))

        audio = self._transform.istft(spec_flat)

        t = audio.shape[-1]
        return mx.reshape(audio, (*batch_dims, 2, t))


def _make_norm(norm_type: Optional[str], channels: int) -> nn.Module:
    """Construct a norm layer that operates on NHWC tensors."""
    if norm_type is None:
        return nn.Identity()
    if norm_type == "BatchNorm":
        return nn.BatchNorm(channels)
    if norm_type == "InstanceNorm":
        # InstanceNorm2d(affine=True) is equivalent to GroupNorm(C, C): with
        # num_groups == channels, mlx.nn.GroupNorm's grouping-order convention
        # (pytorch_compatible or not) is moot -- each group is exactly one
        # channel either way.
        return nn.GroupNorm(channels, channels)
    if norm_type.startswith("GroupNorm"):
        groups = int(norm_type.replace("GroupNorm", ""))
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def _make_act(act_type: str) -> nn.Module:
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    if act_type.startswith("elu"):
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    raise ValueError(f"Unsupported activation: {act_type}")


def _apply_act(act_type: str, x: mx.array) -> mx.array:
    if act_type == "gelu":
        return nn.gelu(x)
    if act_type == "relu":
        return nn.relu(x)
    if act_type.startswith("elu"):
        alpha = float(act_type.replace("elu", ""))
        return nn.elu(x, alpha=alpha)
    raise ValueError(f"Unsupported activation: {act_type}")


def _crop_spatial_to_smallest(a: mx.array, b: mx.array) -> tuple:
    """Crop NHWC tensors to shared H/W shape (handles odd edge effects)."""
    h = min(a.shape[1], b.shape[1])
    w = min(a.shape[2], b.shape[2])
    return a[:, :h, :w, :], b[:, :h, :w, :]


class DownscaleMLX(nn.Module):
    """Norm + activation + strided Conv2d downsample block."""

    def __init__(self, in_c: int, out_c: int, scale: Sequence[int], norm_type: Optional[str], act_type: str):
        super().__init__()
        self.act_type = act_type
        self.norm = _make_norm(norm_type, in_c)
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=tuple(scale),
            stride=tuple(scale),
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = _apply_act(self.act_type, x)
        return self.conv(x)


class UpscaleMLX(nn.Module):
    """Norm + activation + ConvTranspose2d upsample block."""

    def __init__(self, in_c: int, out_c: int, scale: Sequence[int], norm_type: Optional[str], act_type: str):
        super().__init__()
        self.act_type = act_type
        self.norm = _make_norm(norm_type, in_c)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=tuple(scale),
            stride=tuple(scale),
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = _apply_act(self.act_type, x)
        return self.conv(x)


class TfcTdfInnerBlockMLX(nn.Module):
    """One residual TFC+TDF block from MDX23C (NHWC variant)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        freq_bins: int,
        bottleneck_factor: int,
        norm_type: Optional[str],
        act_type: str,
    ):
        super().__init__()
        self.act_type = act_type

        self.tfc1_norm = _make_norm(norm_type, in_c)
        self.tfc1_conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.tdf_norm1 = _make_norm(norm_type, out_c)
        self.tdf_linear1 = nn.Linear(freq_bins, freq_bins // bottleneck_factor, bias=False)
        self.tdf_norm2 = _make_norm(norm_type, out_c)
        self.tdf_linear2 = nn.Linear(freq_bins // bottleneck_factor, freq_bins, bias=False)

        self.tfc2_norm = _make_norm(norm_type, out_c)
        self.tfc2_conv = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self.shortcut(x)

        y = self.tfc1_norm(x)
        y = _apply_act(self.act_type, y)
        y = self.tfc1_conv(y)

        # Apply TDF along the frequency axis (NHWC: axis=2).
        tdf = self.tdf_norm1(y)
        tdf = _apply_act(self.act_type, tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,C,F
        tdf = self.tdf_linear1(tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,F,C
        tdf = self.tdf_norm2(tdf)
        tdf = _apply_act(self.act_type, tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,C,F'
        tdf = self.tdf_linear2(tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,F,C
        y = y + tdf

        y = self.tfc2_norm(y)
        y = _apply_act(self.act_type, y)
        y = self.tfc2_conv(y)

        return y + shortcut


class TfcTdfStackMLX(nn.Module):
    """Stack of residual TFC+TDF inner blocks."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        num_blocks: int,
        freq_bins: int,
        bottleneck_factor: int,
        norm_type: Optional[str],
        act_type: str,
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        current_in = int(in_c)
        for idx in range(self.num_blocks):
            block = TfcTdfInnerBlockMLX(
                in_c=current_in,
                out_c=out_c,
                freq_bins=freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"blocks_{idx}", block)
            current_in = out_c

    def __call__(self, x: mx.array) -> mx.array:
        for idx in range(self.num_blocks):
            block = getattr(self, f"blocks_{idx}")
            x = block(x)
        return x


class TfcTdfV3MLX(nn.Module):
    """MLX implementation of MDX23C TFC_TDF_v3 -- the model this package's
    torch `mdxnet_infer.model.TFC_TDF_net` also implements. Config-driven, so
    the same class serves every registry checkpoint (unlike the sibling
    Roformer package, this package's whole registry shares one architecture;
    only the audio/model hyperparameters in `config` differ per checkpoint --
    see `mdxnet_infer.backends.mlx_backend` module docstring)."""

    def __init__(self, config: Dict):
        super().__init__()

        model_cfg = config["model"]
        audio_cfg = config["audio"]
        training_cfg = config["training"]

        norm_type = model_cfg.get("norm")
        act_type = model_cfg.get("act", "gelu")

        target_instrument = training_cfg.get("target_instrument")
        instruments = training_cfg.get("instruments", [])

        self.num_target_instruments = 1 if target_instrument else len(instruments)
        self.num_subbands = int(model_cfg["num_subbands"])
        self.num_scales = int(model_cfg["num_scales"])
        self.act = _make_act(act_type)

        dim_c = self.num_subbands * int(audio_cfg["num_channels"]) * 2
        scale = tuple(int(v) for v in model_cfg["scale"])
        blocks_per_scale = int(model_cfg["num_blocks_per_scale"])
        channels = int(model_cfg["num_channels"])
        growth = int(model_cfg["growth"])
        bottleneck_factor = int(model_cfg["bottleneck_factor"])
        freq_bins = int(audio_cfg["dim_f"]) // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, channels, kernel_size=1, stride=1, padding=0, bias=False)

        current_channels = channels
        current_freq_bins = freq_bins

        for idx in range(self.num_scales):
            tfc_tdf = TfcTdfStackMLX(
                in_c=current_channels,
                out_c=current_channels,
                num_blocks=blocks_per_scale,
                freq_bins=current_freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            down = DownscaleMLX(
                in_c=current_channels,
                out_c=current_channels + growth,
                scale=scale,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"encoder_blocks_{idx}", nn.Module())
            block = getattr(self, f"encoder_blocks_{idx}")
            block.tfc_tdf = tfc_tdf
            block.downscale = down
            current_freq_bins = current_freq_bins // scale[1]
            current_channels += growth

        self.bottleneck_block = TfcTdfStackMLX(
            in_c=current_channels,
            out_c=current_channels,
            num_blocks=blocks_per_scale,
            freq_bins=current_freq_bins,
            bottleneck_factor=bottleneck_factor,
            norm_type=norm_type,
            act_type=act_type,
        )

        for idx in range(self.num_scales):
            up = UpscaleMLX(
                in_c=current_channels,
                out_c=current_channels - growth,
                scale=scale,
                norm_type=norm_type,
                act_type=act_type,
            )
            current_freq_bins = current_freq_bins * scale[1]
            current_channels -= growth
            tfc_tdf = TfcTdfStackMLX(
                in_c=2 * current_channels,
                out_c=current_channels,
                num_blocks=blocks_per_scale,
                freq_bins=current_freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"decoder_blocks_{idx}", nn.Module())
            block = getattr(self, f"decoder_blocks_{idx}")
            block.upscale = up
            block.tfc_tdf = tfc_tdf

        self.final_conv1 = nn.Conv2d(current_channels + dim_c, current_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.final_conv2 = nn.Conv2d(
            current_channels,
            self.num_target_instruments * dim_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.stft = STFT(
            n_fft=int(audio_cfg["n_fft"]),
            hop_length=int(audio_cfg["hop_length"]),
            dim_f=int(audio_cfg["dim_f"]),
        )

    def cac2cws(self, x: mx.array) -> mx.array:
        """Reshape channel-as-complex to channel-with-subbands space."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = mx.reshape(x, (b, c, k, f // k, t))
        return mx.reshape(x, (b, c * k, f // k, t))

    def cws2cac(self, x: mx.array) -> mx.array:
        """Reverse subband channel reshaping."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = mx.reshape(x, (b, c // k, k, f, t))
        return mx.reshape(x, (b, c // k, f * k, t))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 2, T)
        x = self.stft(x)  # (B, C, F, T)

        mix = x = self.cac2cws(x)  # (B, C', F', T)
        mix_nhwc = mx.transpose(mix, (0, 2, 3, 1))  # (B, F', T, C')

        x = self.first_conv(mix_nhwc)  # (B, F', T, C)
        first_conv_out = x

        # Match PyTorch's transpose(-1, -2) before the encoder/decoder stack.
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, T, F', C)

        encoder_outputs = []
        for idx in range(self.num_scales):
            block = getattr(self, f"encoder_blocks_{idx}")
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for dec_idx in range(self.num_scales):
            block = getattr(self, f"decoder_blocks_{dec_idx}")
            x = block.upscale(x)
            skip = encoder_outputs.pop()
            x, skip = _crop_spatial_to_smallest(x, skip)
            x = mx.concatenate([x, skip], axis=-1)
            x = block.tfc_tdf(x)

        x = mx.transpose(x, (0, 2, 1, 3))  # (B, F', T, C)

        x, first_conv_out = _crop_spatial_to_smallest(x, first_conv_out)
        x = x * first_conv_out  # learned mask, applied as in the torch model

        x, mix_nhwc = _crop_spatial_to_smallest(x, mix_nhwc)
        x = mx.concatenate([mix_nhwc, x], axis=-1)
        x = self.final_conv1(x)
        x = self.act(x)
        x = self.final_conv2(x)

        x = mx.transpose(x, (0, 3, 1, 2))  # (B, C', F', T)
        x = self.cws2cac(x)  # (B, C, F, T)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = mx.reshape(x, (b, self.num_target_instruments, -1, f, t))

        return self.stft.inverse(x)
