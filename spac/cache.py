import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda

from .type_alias import (
    CACHE_ENTRY_NAMES,
    SEQUENCE_ENTRY_NAMES,
    CacheEntryNames,
    SequenceEntryNames,
)

logger = logging.getLogger(__name__)


class SpaCache:

    @classmethod
    def set_hparams(
        cls,
        proxy_projs: torch.nn.ModuleDict,
        update_ratio: np.ndarray,
    ) -> None:
        cls.update_ratio = update_ratio
        cls._proxy_projs = proxy_projs
        cls._proxy_rank = proxy_projs["0"].weight.shape[0]
        cls._cache: Dict[
            CacheEntryNames, Dict[SequenceEntryNames, Dict[int, torch.Tensor]]
        ]
        cls._cache = {
            name: {subseq: {} for subseq in SEQUENCE_ENTRY_NAMES}
            for name in CACHE_ENTRY_NAMES
        }
        logger.info(f"Average update ratio: {update_ratio.mean()}\n{update_ratio}")

    @classmethod
    def reset_cache(
        cls,
        sequence_entry_names: List[SequenceEntryNames] = SEQUENCE_ENTRY_NAMES,
        cache_entry_names: List[CacheEntryNames] = CACHE_ENTRY_NAMES,
        empty_cache: bool = False,
    ) -> None:
        if not hasattr(cls, "_cache"):
            return
        # Cache status
        for name in cache_entry_names:
            for subseq in sequence_entry_names:
                cls._cache[name][subseq] = {}

        if empty_cache:
            cuda.empty_cache()  # included in reset_rope_cache

    @classmethod
    def mark_finished(
        cls,
        finished_batch_idx: torch.Tensor,
        sequence_entry_names: List[SequenceEntryNames] = SEQUENCE_ENTRY_NAMES,
        cache_entry_names: List[CacheEntryNames] = CACHE_ENTRY_NAMES,
    ) -> None:
        if not hasattr(cls, "_cache"):
            return
        for name in cache_entry_names:
            for subseq in sequence_entry_names:
                # all layers
                for layer, value in cls._cache[name][subseq].items():
                    if value is not None:
                        cls._cache[name][subseq][layer] = value[~finished_batch_idx]

    @staticmethod
    def _get_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        return F.cosine_similarity(a, b, dim=dim)

    @classmethod
    def check_update(
        cls,
        attn_input: torch.Tensor,
        layer_id: int,
        prompt_length: int,
        new_indicator: Optional[torch.Tensor] = None,
    ) -> Optional[torch.LongTensor]:
        """
        L >= L_check >= L_update = int(L_check * update_ratio)

        Args:
            attn_input (torch.Tensor): 
                - (B, prompt_length + L, D) if refresh
                - (B, L, D) if cache exists
            layer_id (int): Layer index
            prompt_length (int): Prompt length

        Returns:
            Optional[torch.LongTensor]:
                - None if refresh or refresh_gen
                - (B, L_update) if update
        """
        s_proj = cls._proxy_projs[str(layer_id)]
        refresh = (layer_id not in cls._cache["ff_output"]["prompt"])
        if layer_id == 0:
            return None  # always update at the first layer

        if layer_id not in cls._cache["update_indicator"]["gen"]:
            if refresh:
                cls._cache["update_indicator"]["gen"][layer_id] = (
                    s_proj(attn_input[:, prompt_length:])
                ) if new_indicator is None else new_indicator[:, prompt_length:]
            else:
                # refresh_gen
                cls._cache["update_indicator"]["gen"][layer_id] = (
                    s_proj(attn_input) if new_indicator is None else new_indicator)
            return None

        # elif check_indices is None: # check all
        else:
            update_ratio = cls.update_ratio[int(layer_id)]
            if update_ratio == 1.0:
                update_ratio = None  # always update
            if not (0.0 < update_ratio < 1.0) and update_ratio is not None:
                raise ValueError(
                    f"`update_ratio` should be in (0.0, 1.0). Found {update_ratio}."
                )

            update_length = int(attn_input.shape[1] * update_ratio)
            new_indicator = s_proj(attn_input) if new_indicator is None else new_indicator
            # new_indicator = F.normalize(s_proj(attn_input), p=2, dim=-1)
            old_indicator = cls._cache["update_indicator"]["gen"][layer_id]
            cos_sim = cls._get_similarity(new_indicator, old_indicator, dim=-1)
            update_indices = torch.topk(
                cos_sim,  # (B, L)
                k=update_length,
                dim=-1,
                largest=False,
                sorted=False,  # skip the sorting for performance
            ).indices  # (B, L_update)

            # update cache
            cls._cache["update_indicator"]["gen"][layer_id] = new_indicator
            return update_indices

    @classmethod
    def get_attn_input(
        cls,
        attn_input: torch.Tensor,
        attn_input_normed: torch.Tensor,
        update_indices: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if update_indices is None:
            return attn_input, attn_input_normed
        else:
            update_indices = update_indices.unsqueeze(-1).expand(-1, -1, attn_input.shape[-1])
            return (
                torch.gather(attn_input, dim=1, index=update_indices),
                torch.gather(attn_input_normed, dim=1, index=update_indices),
            )

    @classmethod
    def get_update_tokens(
        cls,
        update_indices: Optional[torch.LongTensor],
        *feature_sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        if update_indices is None:
            return feature_sequences
        else:
            update_indices = update_indices.unsqueeze(-1).expand(-1, -1, feature_sequences[0].shape[-1])
            return (torch.gather(feature, dim=1, index=update_indices) for feature in feature_sequences)

    @classmethod
    def get_update_tokens_h(
        cls,
        update_indices: Optional[torch.LongTensor],
        attn_input: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        if update_indices is None:
            return attn_input
        else:
            update_indices = update_indices.unsqueeze(-1).expand(-1, -1, attn_input.shape[-1])
            return torch.gather(attn_input, dim=1, index=update_indices)

    @classmethod
    def get_update_tokens_q(
        cls,
        update_indices: Optional[torch.LongTensor],
        q: torch.Tensor,
    ) -> torch.Tensor:
        if update_indices is None:
            return q
        else:
            update_indices = update_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.shape[1], -1, q.shape[-1])
            return torch.gather(q, dim=2, index=update_indices)

    @classmethod
    def get_pos_emb(
        cls,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_id: int,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_id == 0:
            return cos, sin  # always update at the first layer

        refresh = (layer_id not in cls._cache["ff_output"]["prompt"])
        if refresh:
            return cos, sin
        else:  # refresh_gen or update
            return (
                cos[:, prompt_length:],
                sin[:, prompt_length:],
            )
    
    @classmethod
    def get_attn_mask(
        cls,
        attn_mask: Optional[torch.BoolTensor],
        layer_id: int,
        prompt_length: int,
        update_indices: Optional[torch.LongTensor],
    ) -> Optional[torch.Tensor]:
        if layer_id == 0:
            return attn_mask  # always update at the first layer

        if not isinstance(attn_mask, torch.Tensor):
            return attn_mask

        refresh = (layer_id not in cls._cache["ff_output"]["prompt"])
        if update_indices is None:
            if refresh:
                return attn_mask
            else:  # refresh_gen
                # attn_mask # (B, 1, prompt_length + L, prompt_length + L)
                return attn_mask[:, :, prompt_length:]
        else:  # update
            assert not refresh
            update_indices = update_indices[:, None, :, None] # (B, 1, L_update, 1)
            return torch.gather(
                attn_mask[:, :, prompt_length:], # (B, 1, L, prompt_length + L)
                dim=2,
                index=update_indices.expand(-1, 1, -1, attn_mask.shape[-1]),
            )

    @classmethod
    def get_kv_heads(
        cls,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        prompt_length: int,
        update_indices: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k, v (torch.Tensor):
                - (B, H, prompt_length + L, D) if refresh
                - (B, H, L, D) if refresh_gen
                - (B, H, L_update, D) if update
            layer_id (int): Layer index
            prompt_length (int): Prompt length
            update_indices (Optional[torch.LongTensor]): Update indices
                - None if refresh or refresh_gen
                - (B, L_update) if update

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, H, prompt_length + L, D)
        """
        if layer_id == 0:
            return k, v  # always update at the first layer

        refresh = (layer_id not in cls._cache["ff_output"]["prompt"])
        if update_indices is None:
            # assert layer_id not in cls._cache["k"]["gen"]
            # assert layer_id not in cls._cache["v"]["gen"]

            if refresh:
                cls._cache["k"]["prompt"][layer_id] = k
                cls._cache["v"]["prompt"][layer_id] = v
                return k, v

            else: # refresh_gen
                cls._cache["k"]["prompt"][layer_id][:, :, prompt_length:] = k
                cls._cache["v"]["prompt"][layer_id][:, :, prompt_length:] = v
        else: # update
            assert not refresh
            cls._cache["k"]["prompt"][layer_id][:, :, prompt_length:] = k
            cls._cache["v"]["prompt"][layer_id][:, :, prompt_length:] = v

        return (
            cls._cache["k"]["prompt"][layer_id],
            cls._cache["v"]["prompt"][layer_id],
        )

    @classmethod
    def get_ff_output(
        cls,
        ff_output: torch.Tensor,
        attn_output: torch.Tensor,
        attn_input: torch.Tensor,
        layer_id: int,
        prompt_length: int,
        update_indices: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Args:
            ff_output (torch.Tensor):
                - (B, H, prompt_length + L, D) if refresh
                - (B, H, L, D) if refresh_gen
                - (B, H, L_update, D) if update
            attn_output (torch.Tensor):
                - (B, H, prompt_length + L, D) if refresh
                - (B, H, L, D) if refresh_gen
                - (B, H, L_update, D) if update
            attn_input (torch.Tensor): 
                - (B, H, prompt_length + L, D) if refresh 
                - (B, H, L, D) otherwise
            layer_id (int): Layer index
            prompt_length (int): Prompt length
            update_indices (Optional[torch.LongTensor]):
                - None if refresh or refresh_gen
                - (B, L_update) if update

        Returns:
            torch.Tensor:
                - (B, H, prompt_length + L, D) if refresh
                - (B, H, L, D) if refresh_gen
                - (B, H, L, D) if update
        """
        refresh = (layer_id not in cls._cache["ff_output"]["prompt"])

        ff_output = ff_output + attn_output # ff_output after residual connection
        if update_indices is None:
            if refresh:
                assert layer_id not in cls._cache["ff_output"]["gen"]
                cls._cache["ff_output"]["prompt"][layer_id] = None
                cls._cache["ff_output"]["gen"][layer_id] = ff_output[:, prompt_length:]
            elif layer_id == 0: # TODO: possible bug for layer 0
                return (ff_output + attn_input)[:, prompt_length:]
            else: # refresh_gen
                # assert layer_id not in cls._cache["ff_output"]["gen"] # comment out for benchmarking
                cls._cache["ff_output"]["gen"][layer_id] = ff_output
            return ff_output + attn_input

        else: # update
            assert not refresh

            cls._cache["ff_output"]["gen"][layer_id].scatter_(
                dim=1,
                index=update_indices.unsqueeze(-1).expand(-1, -1, ff_output.shape[2]),
                src=ff_output,
            )

            return (
                cls._cache["ff_output"]["gen"][layer_id] +
                attn_input
            )
        


def _split_gaussian_frequency(
    layer,
    n_layers,
    peak_pos=0.75, 
    freq_peak=0.25,
    freq_start=0.01, 
    freq_end=0.15,
    freq_min=0.0,
    scale=1.0,
):
    # Normalize
    t = np.array(layer) / n_layers
    freq_peak = freq_peak * scale
    freq_start = freq_start * scale
    freq_end = freq_end * scale

    mu = peak_pos

    # Sanity checks to prevent math errors
    if freq_start >= freq_peak or freq_end >= freq_peak:
        raise ValueError("Boundary values (freq_start, freq_end) must be strictly less than freq_peak.")

    # 3. Solve for Sigmas (Widths) analytically
    # sigma = sqrt( -(x-mu)^2 / (2 * ln(y/A)) )
    # Avoid log(0) by clamping slightly above 0
    sigma_L = np.sqrt(-(0 - mu)**2 / (2 * np.log(max(freq_start, 1e-9) / freq_peak)))
    sigma_R = np.sqrt(-(1 - mu)**2 / (2 * np.log(max(freq_end, 1e-9) / freq_peak)))
    
    # 4. Generate Curve
    values = np.zeros_like(t, dtype=float)
    left_mask = t < mu
    right_mask = t >= mu
    
    # Compute Left side
    # values[left_mask] = (freq_peak * np.exp( -((t[left_mask] - mu)**2) / (2 * sigma_L**2) ))
    values[left_mask] = (freq_peak * np.exp( -((t[left_mask] - mu)**2) / (2 * sigma_L**2) )).clip(freq_min, None)
    
    # Compute Right side
    values[right_mask] = freq_peak * np.exp( -((t[right_mask] - mu)**2) / (2 * sigma_R**2) )
    # values[right_mask] = (freq_peak * np.exp( -((t[right_mask] - mu)**2) / (2 * sigma_R**2) )).clip(freq_min, None)
    
    return values


def get_update_ratio(
    pretrained_model_name_or_path: str,
    freq_dist: Literal['gaussian', 'uniform'],
    max_update_ratio: Optional[float] = None,
    avg_update_ratio: Optional[float] = None,
    min_update_ratio: float = 2 ** -5,
):
    NAME_TO_KWARGS = {
        'LLaDA-8B-Instruct': dict(
            n_layers=32, peak_pos=0.75, freq_start=0.03, freq_peak=0.25, freq_end=0.13),
        'LLaDA-1.5': dict(n_layers=32, peak_pos=0.8, freq_start=0.03, freq_peak=0.25, freq_end=0.13),
        'Dream-v0-Instruct-7B': dict(n_layers=28, peak_pos=0.5, freq_start=0.05, freq_peak=0.3, freq_end=0.25),
    }
    pretrained_model_name_or_path = pretrained_model_name_or_path.split('/')[-1]
    if pretrained_model_name_or_path not in NAME_TO_KWARGS:
        raise ValueError(
            f"Pretrained model {pretrained_model_name_or_path} not supported for automatic update ratio calculation. "
            f"Supported models: {list(NAME_TO_KWARGS.keys())}."
        )
    kwargs = NAME_TO_KWARGS[pretrained_model_name_or_path]

    if freq_dist == 'uniform':
        return np.array([max_update_ratio or avg_update_ratio] * kwargs['n_layers'])
    elif freq_dist == 'gaussian':
        if max_update_ratio is not None:
            scale = max_update_ratio / kwargs['freq_peak']
            if avg_update_ratio is not None:
                logger.warning(
                    "Both max_update_ratio and avg_update_ratio are provided. "
                    f"Using {max_update_ratio=} for gaussian frequency distribution."
                    f" Ignoring {avg_update_ratio=}."
                )
        elif avg_update_ratio is not None:
            cur_update_ratio = _split_gaussian_frequency(
                layer=np.arange(kwargs['n_layers']),
                **kwargs,
            ).mean()
            scale = avg_update_ratio / cur_update_ratio
        else:
            raise ValueError(
                "Either max_update_ratio or avg_update_ratio must be provided for gaussian frequency distribution."
            )
        return _split_gaussian_frequency(
            layer=np.arange(kwargs['n_layers']),
            scale=scale,
            freq_min=min_update_ratio,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"freq_dist should be 'gaussian' or 'uniform'. Found {freq_dist}."
        )
