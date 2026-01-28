from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torch.distributions as dists
import torch.nn.functional as F
from transformers.utils import ModelOutput

from ..sequence import SequenceMgmt
from ..utils import BenchmarkLogger

logger = getLogger(__name__)


# Copied from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py
@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    ttft: Optional[float] = None


# Copied from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


# Copied from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py
def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


# Copied from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


# Copied from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py
@torch.no_grad()
def sample(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config,
    generation_tokens_hook_func,
    generation_logits_hook_func,
    log_ttft=False,
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    bench_logger = BenchmarkLogger(log_latency=log_ttft, mode='sum')

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    # this allows user-defined token control of the intermediate steps
    x = generation_tokens_hook_func(None, x, None)
    for i in range(steps):
        mask_index = (x == mask_token_id)
        context = bench_logger if i == 0 else nullcontext()
        with context:
            logits = self(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        # this allows user-defined logits control of the intermediate steps
        logits = generation_logits_hook_func(i, x, logits)

        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
    
        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
            _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
            x[mask_index] = x0.clone()
        else:
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(i, x, logits)

        if histories is not None:
            histories.append(x.clone())
    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
            ttft=bench_logger.latency if log_ttft else None, # total ttft, seconds
        )
    else:
        return x


@torch.no_grad()
def sample_spa_cache(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config,
    generation_tokens_hook_func,
    generation_logits_hook_func,
    log_ttft=False,
    **kwargs,
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    if kwargs:
        for k, v in kwargs.items():
            if k not in ["tokenizer"]:
                logger.warning(f"Unexpected keyword argument: {k}={v}")
    # breakpoint()

    # pad input_ids to max_length
    # x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    x = SequenceMgmt.get_model_input(input_ids=input_ids)
    bench_logger = BenchmarkLogger(log_latency=log_ttft, mode='sum')

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    # this allows user-defined token control of the intermediate steps
    x = generation_tokens_hook_func(None, x, None)
    for i in range(steps):
        mask_index = (x == mask_token_id)
        context = bench_logger if i == 0 else nullcontext()
        with context:
            logits = self(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        # this allows user-defined logits control of the intermediate steps
        logits = generation_logits_hook_func(i, x, logits)

        if SequenceMgmt.refresh():
            mask_logits = logits[mask_index]
        else:
            mask_logits = logits[mask_index[:, SequenceMgmt.prompt_length:]]

        t = timesteps[i]
        s = timesteps[i + 1]
    
        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
            _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
            x[mask_index] = x0.clone()
        else:
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            number_transfer_tokens = max(number_transfer_tokens, 1)  # at least transfer 1 token
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(i, x, logits)
        # breakpoint()

        if histories is not None: # TODO: debug ily's issue
            histories.append(x.clone())

        # update x
        x, attention_mask, tok_idx = SequenceMgmt.next_step(
            x, return_attn_mask=True, attn_mask=attention_mask, tok_idx=tok_idx)
        if x is None:  # early stop
            break

    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=SequenceMgmt.output_ids,
            history=histories,
            ttft=bench_logger.latency if log_ttft else None, # total ttft, seconds
        )
    else:
        return x
