#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Created :   2025/08/10 23:36:56
@Desc    :
@Ref     :
    (2025/08/13) Load proxy projections as nn.Linear
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHITECTURES = ("LLaDAModelLM", "DreamModel")


def save_projection_svd_from_model(
    model: AutoModel,
    transformer_blocks_name: str,
    feature_save_dir: Path,
    device: str,
):
    feature_save_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(total=model.config.n_layers, desc="Saving SVD features")
    for name, module in model.named_modules():
        if name == transformer_blocks_name:
            module: nn.ModuleList
            for block in module:
                Wt = block.v_proj.weight.data
                Wt = Wt.to(device, dtype=torch.float32)

                V, S, _ = torch.linalg.svd(Wt.transpose(0, 1), full_matrices=False)
                torch.save(
                    {"V": V, "S": S},
                    feature_save_dir / f"layer_{block.layer_id:02d}.pt",
                )
                progress_bar.update(1)
    progress_bar.close()


def load_proxy_projection_from_dir(
    model_config,
    proxy_rank: int | float,
    feature_save_dir: Path,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> nn.ModuleDict:
    # Load proxy projections from disk
    if model_config.architectures[0] not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported model architecture: {model_config.architectures}, supported: {_SUPPORTED_ARCHITECTURES}"
        )

    if model_config.architectures[0] == "LLaDAModelLM":
        head_dim = model_config.d_model // model_config.n_heads
        kv_proj_in_dim = model_config.d_model
        kv_proj_out_dim = int(model_config.n_kv_heads * head_dim)
        n_layers = model_config.n_layers
    elif model_config.architectures[0] == "DreamModel":
        kv_proj_in_dim = model_config.hidden_size
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        kv_proj_out_dim = int(model_config.num_key_value_heads * head_dim)
        n_layers = model_config.num_hidden_layers

    device = str(device) or ("cuda" if torch.cuda.is_available() else "cpu")

    if proxy_rank > 1:
        proxy_rank = int(proxy_rank)
    elif proxy_rank <= 1:
        proxy_rank = int(kv_proj_out_dim * proxy_rank)

    if proxy_rank < 1:
        raise ValueError(f"Invalid argument {proxy_rank=}")

    logger.info(
        f"Loading proxy projection with rank {proxy_rank} from {feature_save_dir}"
    )
    proxy_projection = {}
    for layer_id in tqdm(range(n_layers)):
        projection_svd = torch.load(
            Path(feature_save_dir) / f"layer_{layer_id:02d}.pt",
            map_location=device,
            weights_only=True,
        )
        weights = (
            projection_svd["V"][:, :proxy_rank]
            * projection_svd["S"][None, :proxy_rank]
        ).T
        projection = nn.Linear(kv_proj_in_dim, proxy_rank, bias=False)
        projection.weight.data.copy_(weights)
        proxy_projection[str(layer_id)] = projection

    # breakpoint()
    proxy_projection = nn.ModuleDict(proxy_projection).to(device, dtype=dtype)
    logger.info(f"Loaded proxy projection for {n_layers} layers")

    return proxy_projection


def load_proxy_projection_from_model(
    model: AutoModel,
    transformer_blocks_name: str,
    proxy_rank: int | float,
    device: str,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    feature_save_dir: Path = Path('./results/svd_cache'),
):
    try:
        return load_proxy_projection_from_dir(
            model.config,
            proxy_rank,
            feature_save_dir,
            device,
            dtype,
        )
    except FileNotFoundError:
        logger.warning(
            f"SVD features not found in {feature_save_dir}, generating from model..."
        )
        save_projection_svd_from_model(
            model,
            transformer_blocks_name,
            feature_save_dir,
            device,
        )
        return load_proxy_projection_from_dir(
            model.config,
            proxy_rank,
            feature_save_dir,
            device,
            dtype,
        )