from typing import Tuple

import torch
import torch.nn as nn


def get_rotary_embedding(
    module: nn.Module,
    prefix_len: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        (pos_sin := module._RotaryEmbedding__cache.get("rope_pos_sin")) is not None
        and (pos_cos := module._RotaryEmbedding__cache.get("rope_pos_cos")) is not None
        and pos_sin.shape[-2] >= prefix_len + seq_len
        and pos_cos.shape[-2] >= prefix_len + seq_len
    ):
        if pos_sin.device != device:
            pos_sin = pos_sin.to(device)
            module._RotaryEmbedding__cache["rope_pos_sin"] = pos_sin
        if pos_cos.device != device:
            pos_cos = pos_cos.to(device)
            module._RotaryEmbedding__cache["rope_pos_cos"] = pos_cos
        if prefix_len == 0:
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
        return (
            pos_sin[:, :, prefix_len : (prefix_len + seq_len), :],
            pos_cos[:, :, prefix_len : (prefix_len + seq_len), :],
        )

    with torch.autocast(device.type, enabled=False):
        dim = module.config.d_model // module.config.n_heads
        inv_freq = 1.0 / (
            module.rope_theta
            ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        )
        seq = torch.arange(prefix_len + seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        pos_sin, pos_cos = (
            positions.sin()[None, None, :, :],
            positions.cos()[None, None, :, :],
        )
    module._RotaryEmbedding__cache["rope_pos_sin"] = pos_sin
    module._RotaryEmbedding__cache["rope_pos_cos"] = pos_cos
    if prefix_len == 0:
        return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
    return (
        pos_sin[:, :, prefix_len : (prefix_len + seq_len), :],
        pos_cos[:, :, prefix_len : (prefix_len + seq_len), :],
    )