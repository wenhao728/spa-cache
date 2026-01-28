#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Created :   2025/09/03 21:16:19
@Desc    :
@Ref     :
"""
import logging
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..sequence import SequenceMgmt
from ..utils import BenchmarkLogger

logger = logging.getLogger(__name__)


# Copied from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# Copied from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
def get_num_transfer_tokens(
    mask_index,
    steps,
    num_masks_per_block=None,
    batch_size=None,
    device=None,
):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    batch_size = batch_size or mask_index.size(0)
    if num_masks_per_block is not None:
        mask_num = torch.full(
            (batch_size, 1), num_masks_per_block, dtype=torch.int, device=device
        )
    else:
        mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            batch_size, steps, device=device or mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(batch_size):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


# Copied from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    log_ttft=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    context = nullcontext()

    with context:
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long
        ).to(model.device)
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    bench_logger = BenchmarkLogger(log_latency=log_ttft)

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        with context as prof:
            for i in range(steps):
                mask_index = x == mask_id

                context = bench_logger if num_block == 0 and i == 0 else nullcontext()
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    with context:
                        logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # b, l
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

    if log_ttft:
        return x, bench_logger.latency
    else:
        return x

@torch.no_grad()
def generate_spa_cache(
    model,
    prompt,
    block_length=128,
    steps=128,
    temperature=0.0,
    remasking="low_confidence",
    log_ttft=False,
    threshold=None,
    dynamic_threshold_factor=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kwargs:
        for k, v in kwargs.items():
            if k not in ["tokenizer"]:
                logger.warning(f"Unexpected keyword argument: {k}={v}")


    # Reset step counter
    x = SequenceMgmt.get_model_input(input_ids=prompt)
    num_blocks = SequenceMgmt.gen_length // block_length
    steps = steps // num_blocks

    bench_logger = BenchmarkLogger(log_latency=log_ttft)

    for num_block in range(num_blocks):
        if x is None: break
        block_start = num_block * block_length + SequenceMgmt.prompt_length # input always has prompt
        # block_start = num_block * block_length
        # if SequenceMgmt.refresh():
        #     block_start += SequenceMgmt.prompt_length
        block_end = block_start + block_length
        block_mask_index = x[:, block_start:block_end] == SequenceMgmt.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            block_start = num_block * block_length + SequenceMgmt.prompt_length # input always has prompt
            # block_start = (num_block + 1) * block_length
            # if SequenceMgmt.refresh():
            #     block_start += SequenceMgmt.prompt_length
            block_end = block_start + block_length

            if x is None: break
            mask_index = (x == SequenceMgmt.mask_id)
            mask_index[:, block_end:] = False
            if mask_index[:, block_start:block_end].sum() == 0:
                break

            context = bench_logger if num_block == 0 and i == 0 else nullcontext()
            with context:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                confidence = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                confidence = torch.rand(
                    (x0.shape[0], x0.shape[1]), device=x0.device
                )
            else:
                raise NotImplementedError(remasking)

            # mask the suffix tokens
            block_end_ = (num_block + 1) * block_length
            if SequenceMgmt.refresh():
                block_end_ += SequenceMgmt.prompt_length
            confidence[:, block_end_ :] = -np.inf
            # mask the decoded tokens (current and previous blocks)
            if SequenceMgmt.refresh():
                confidence = torch.where(mask_index, confidence, -np.inf)
                x0 = torch.where(mask_index, x0, x)
            else:
                confidence = torch.where(
                    mask_index[:, SequenceMgmt.prompt_length :], confidence, -np.inf)
                x0 = torch.where(
                    mask_index[:, SequenceMgmt.prompt_length :], x0, x[:, SequenceMgmt.prompt_length :])

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )

            # for j in range(confidence.shape[0]):
            #     _, select_index = torch.topk(
            #         confidence[j], k=num_transfer_tokens[j, i],
            #     )
            #     transfer_index[j, select_index] = True
            #     if return_intermediate:
            #         selected_indices.append(
            #             (select_index - SequenceMgmt.prompt_length) 
            #             if SequenceMgmt.refresh() else select_index
            #         )  # the prompts are not truncated
            if dynamic_threshold_factor is not None:
                transfer_index = get_num_transfer_indices_v6_dynamic_threshold(
                    confidence, transfer_index, 
                    mask_index if SequenceMgmt.refresh() else mask_index[:, SequenceMgmt.prompt_length :],
                    dynamic_threshold_factor,
                )
            elif threshold is not None:
                transfer_index = get_transfer_indices_v6_threshold(
                    confidence, transfer_index, 
                    mask_index if SequenceMgmt.refresh() else mask_index[:, SequenceMgmt.prompt_length :],
                    threshold,
                )
            else:
                transfer_index = get_transfer_indices_v6(
                    confidence, transfer_index, num_transfer_tokens[:, i])

            if SequenceMgmt.refresh():
                x[transfer_index] = x0[transfer_index]
            else:
                x[:, SequenceMgmt.prompt_length :][transfer_index] = x0[transfer_index]

            # Increment step
            # logger.debug(
            #     "[before truncate] x0={}".format(
            #         kwargs["tokenizer"].batch_decode(
            #             x0[-1, (SequenceMgmt.prompt_length if SequenceMgmt.refresh() else 0) :]
            #         )
            #     )
            # )
            # logger.debug(
            #     "[before truncate] x={}".format(
            #         kwargs["tokenizer"].batch_decode(
            #             x[-1, (SequenceMgmt.prompt_length if SequenceMgmt.refresh() else 0) :]
            #         )
            #     )
            # )
            # x = SequenceMgmt.next_step(x, transfer_index=torch.cat(transfer_index_, dim=0).view(-1, 1))
            x = SequenceMgmt.next_step(x)
            # logger.debug(
            #     "[after truncate] x={}".format(
            #         kwargs["tokenizer"].batch_decode(x[-1, :])
            #     )
            # )
            # breakpoint()

    if log_ttft:
        return SequenceMgmt.output_ids, bench_logger.latency
    else:
        return SequenceMgmt.output_ids


def get_transfer_indices_v6(confidence, transfer_index, num_transfer_tokens):
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(
            confidence[j], k=num_transfer_tokens[j],
        )
        transfer_index[j, select_index] = True
    return transfer_index


def get_transfer_indices_v6_threshold(confidence, transfer_index, mask_index, threshold):
    assert threshold is not None

    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(
            confidence[j], k=num_transfer_tokens[j]
        )
        transfer_index[j, select_index] = True
        
        for k in range(1, num_transfer_tokens[j]):
            if confidence[j, select_index[k]] < threshold:
                transfer_index[j, select_index[k]] = False
    return transfer_index


def get_num_transfer_indices_v6_dynamic_threshold(
    confidence, transfer_index, mask_index, factor,
):
    assert factor is not None

    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        if len(threshs) == 0:
            continue

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True
    return transfer_index
